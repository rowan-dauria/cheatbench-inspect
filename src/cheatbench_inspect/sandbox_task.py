import asyncio
import hashlib
import json
import tarfile
import tempfile
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any

from anyio import CancelScope
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
from inspect_scout import messages_as_str

from cheatbench_inspect.harnesses.container import (
    AGENT_ENV,
    AGENT_USER,
    STREAM_BOOTSTRAP,
    hand_staged_files_to_agent,
)
from cheatbench_inspect.harnesses.harness import (
    CLI_HARNESSES,
    HarnessPolicy,
    HarnessResult,
    apply_harness_result,
    run_harness,
)
from cheatbench_inspect.judge import build_judge_prompt


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
async def sandbox_service_endpoint(
    config: dict[str, Any],
    *,
    grader_state: dict[str, Any] | None = None,
    startup_timeout_s: float = 30,
    user: str = AGENT_USER,
) -> AsyncGenerator[str, None]:
    """Run a sandbox service that reports its URL on the first stdout line."""
    process = await sandbox().exec_remote(
        ["python3", "-I", "-c", STREAM_BOOTSTRAP],
        ExecRemoteStreamingOptions(
            input=json.dumps(config) + "\n",
            stdin_open=True,
            user=user,
            env=AGENT_ENV if user == AGENT_USER else None,
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
                        f"Sandbox service exited before reporting its URL: {stderr.strip()}"
                    )
                if isinstance(event, ExecStderr):
                    stderr += event.data
                elif isinstance(event, ExecStdout):
                    stdout += event.data
        url, stdout = stdout.split("\n", 1)
        launch_error = None
        try:
            yield url
        except BaseException as error:
            launch_error = error
            raise
        finally:
            if grader_state is not None:
                try:
                    # A bridge limit can cancel the enclosing Inspect task group.
                    with CancelScope(shield=True):
                        async with asyncio.timeout(30):
                            await process.close_stdin()
                            while (event := await anext(events, None)) is not None:
                                if isinstance(event, ExecStdout):
                                    stdout += event.data
                                elif isinstance(event, ExecStderr):
                                    stderr += event.data
                            if not stdout.strip():
                                raise RuntimeError(
                                    f"Verifier did not report its state: {stderr.strip()}"
                                )
                            grader_state.update(json.loads(stdout))
                except Exception as error:
                    if launch_error is None:
                        raise
                    launch_error.add_note(f"Verifier state collection failed: {error}")
    finally:
        with CancelScope(shield=True):
            await process.kill()


sandbox_verifier_endpoint = sandbox_service_endpoint


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
    restore_staged_atimes: bool = False,
    agent_env: dict[str, str] | None = None,
    stage_files: Callable[[list[str]], Awaitable[None]] | None = None,
    launch_harness: Callable[..., Awaitable[HarnessResult | AgentState]] | None = None,
) -> Solver:
    stage = stage_files or hand_staged_files_to_agent
    launch = launch_harness or run_harness

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await stage(state.metadata["staged_files"])
        for path, stamp in state.metadata.get("file_mtimes", {}).items():
            time_flags = ["-am"] if restore_staged_atimes else ["-m"]
            result = await sandbox().exec(
                ["touch", "-c", *time_flags, "-d", stamp, "--", path],
                env={**AGENT_ENV, "TZ": "UTC"},
                user=AGENT_USER,
            )
            if not result.success:
                raise RuntimeError(f"Could not set staged file mtime for {path}: {result.stderr}")
        stock_agent = harness in CLI_HARNESSES
        grader_state: dict[str, Any] | None = (
            {}
            if verifier_endpoint is not None and (stock_agent or collect_terminus_grader_state)
            else None
        )
        endpoint = verifier_endpoint(state, grader_state) if verifier_endpoint else nullcontext()
        # Terminus-2 and Kimi Code enforce the original's per-episode step limit
        # natively; Kimi's CLI reads it from KIMI_LOOP_MAX_STEPS_PER_TURN, as the
        # original's [loop_control].max_steps_per_turn. inspect_swe gives Claude Code,
        # Codex and Gemini CLI no way to set one, so they rely on the task's
        # message_limit alone (PR #10 decision).
        try:
            async with endpoint as url:
                policy = HarnessPolicy(
                    mcp_servers=(
                        [{"name": verifier_name, "transport": "streamable-http", "url": url}]
                        if url is not None
                        else []
                    ),
                    max_turns=max_turns if not stock_agent or harness == "kimi-code" else None,
                    timeout_s=timeout_s,
                    env=agent_env or {},
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
        finally:
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
            trajectory=await messages_as_str(state.messages),
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
