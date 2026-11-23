import asyncio
import hashlib
import json
import tarfile
import tempfile
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any

from inspect_ai.agent import AgentState
from inspect_ai.model import GenerateConfig, Model, ResponseSchema
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import Generate, Solver, TaskState
from inspect_ai.util import (
    ExecRemoteStreamingOptions,
    ExecStderr,
    ExecStdout,
    json_schema,
    sandbox,
    time_limit,
)
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH
from inspect_evals_utils.huggingface import hf_hub_download

from cheatbench_inspect.harnesses.container import (
    AGENT_ENV,
    AGENT_USER,
    STREAM_BOOTSTRAP,
    hand_staged_files_to_agent,
)
from cheatbench_inspect.harnesses.harness import (
    HarnessPolicy,
    HarnessResult,
    apply_harness_result,
    run_harness,
)
from cheatbench_inspect.judge import build_judge_prompt, chat_history


def unpack_archive(archive: Path, task_dir: Path) -> None:
    """Unpack the archive so that task_dir appears complete or not at all."""
    task_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=task_dir.parent) as staging:
        unpacked = Path(staging) / "data"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")
        try:
            unpacked.rename(task_dir)
        except OSError:
            # Another process unpacked the same archive first.
            if not (task_dir / "data.jsonl").is_file():
                raise


def fetch_task_dir(
    repo_id: str,
    filename: str,
    revision: str,
    sha256: str,
    cache_subdirectory: str,
    cache_root: Path = INSPECT_EVALS_CACHE_PATH,
    download: Callable[..., str] | None = None,
) -> Path:
    task_dir = cache_root / "cheatbench_inspect" / cache_subdirectory
    task_dir /= sha256[:12]
    if (task_dir / "data.jsonl").is_file():
        return task_dir

    archive = Path(
        (download or hf_hub_download)(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{filename} has sha256 {digest}, expected {sha256}")
    unpack_archive(archive, task_dir)
    return task_dir


@asynccontextmanager
async def sandbox_verifier_endpoint(
    config: dict[str, Any],
    *,
    grader_state: dict[str, Any] | None = None,
    startup_timeout_s: float = 30,
) -> AsyncGenerator[str, None]:
    """Run an MCP verifier in the sandbox and optionally collect its final state."""
    process = await sandbox().exec_remote(
        ["python3", "-I", "-c", STREAM_BOOTSTRAP],
        ExecRemoteStreamingOptions(
            input=json.dumps(config) + "\n", stdin_open=True, user=AGENT_USER, env=AGENT_ENV
        ),
    )
    try:
        # Keep the single-use iterator open through the state snapshot.
        events = aiter(process)
        stdout = ""
        stderr = ""
        async with asyncio.timeout(startup_timeout_s):
            while "\n" not in stdout:
                event = await anext(events, None)
                if event is None:
                    raise RuntimeError(
                        f"Verifier exited before reporting its URL: {stderr.strip()}"
                    )
                if isinstance(event, ExecStderr):
                    stderr += event.data
                elif isinstance(event, ExecStdout):
                    stdout += event.data
        url, stdout = stdout.split("\n", 1)
        yield url
        if grader_state is not None:
            async with asyncio.timeout(30):
                await process.close_stdin()
                while (event := await anext(events, None)) is not None:
                    if isinstance(event, ExecStdout):
                        stdout += event.data
                    elif isinstance(event, ExecStderr):
                        stderr += event.data
                if not stdout.strip():
                    raise RuntimeError(f"Verifier did not report its state: {stderr.strip()}")
                grader_state.update(json.loads(stdout))
    finally:
        await asyncio.shield(process.kill())


def sandbox_solver(
    harness: str,
    harness_version: str | None,
    harness_model: str | None,
    max_turns: int,
    timeout_s: int,
    verifier_endpoint: Callable[
        [TaskState, dict[str, Any] | None], AbstractAsyncContextManager[str]
    ]
    | None = None,
    *,
    verifier_name: str = "verifier",
    collect_terminus_grader_state: bool = False,
    stage_files: Callable[[list[str]], Awaitable[None]] | None = None,
    launch_harness: Callable[..., Awaitable[HarnessResult | AgentState]] | None = None,
) -> Solver:
    stage = stage_files or hand_staged_files_to_agent
    launch = launch_harness or run_harness

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await stage(state.metadata["staged_files"])
        for path, stamp in state.metadata.get("file_mtimes", {}).items():
            result = await sandbox().exec(
                ["touch", "-c", "-m", "-d", stamp, "--", path],
                env={**AGENT_ENV, "TZ": "UTC"},
                user=AGENT_USER,
            )
            if not result.success:
                raise RuntimeError(f"Could not set staged file mtime for {path}: {result.stderr}")
        stock_agent = harness in {"claude-code", "codex"}
        grader_state: dict[str, Any] | None = (
            {}
            if verifier_endpoint is not None and (stock_agent or collect_terminus_grader_state)
            else None
        )
        endpoint = verifier_endpoint(state, grader_state) if verifier_endpoint else nullcontext()
        async with endpoint as url:
            policy = HarnessPolicy(
                mcp_servers=(
                    [{"name": verifier_name, "transport": "streamable-http", "url": url}]
                    if url is not None
                    else []
                ),
                max_turns=None if stock_agent else max_turns,
                timeout_s=timeout_s,
            )
            options = (
                {"cli_version": harness_version or "", "cli_model": harness_model or ""}
                if stock_agent
                else None
            )
            with time_limit(timeout_s if stock_agent else None):
                result = await launch(harness, state, policy, options)
        if isinstance(result, HarnessResult):
            apply_harness_result(state, result)
        if grader_state is not None:
            artifacts = state.store.get("harness_artifacts") or {}
            state.store.set("harness_artifacts", {**artifacts, "grader_state": grader_state})
        return state

    return solve


def trajectory_scorer(
    verdict_model: type[Any],
    extra_context: str,
    judge: Model,
    *,
    explanation_field: str = "reasoning",
) -> Callable[[TaskState, Target], Awaitable[Score]]:
    async def score(state: TaskState, target: Target) -> Score:
        judge_prompt = build_judge_prompt(
            extra_context=extra_context,
            trajectory=chat_history(state),
            agent_prompt=state.input_text,
        )

        judge_response = await judge.generate(
            input=judge_prompt,
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    # Keep the original schema name and the port's description.
                    name="SharedVerdict",
                    description="SharedVerdict",
                    json_schema=json_schema(verdict_model),
                    strict=True,
                )
            ),
        )

        verdict = verdict_model.model_validate_json(judge_response.completion)
        metadata = verdict.model_dump()
        if error := state.store.get("harness_error"):
            metadata["harness_error"] = error
        return Score(
            value=verdict.cheated,
            explanation=getattr(verdict, explanation_field),
            metadata=metadata,
        )

    return score
