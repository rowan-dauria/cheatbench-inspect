"""Launch the SDK episode inside the sample sandbox."""

import json
from pathlib import Path, PurePosixPath
from typing import Any

from inspect_ai.agent import sandbox_agent_bridge
from inspect_ai.model import get_model
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from cheatbench_inspect.container import AGENT_ENV, AGENT_USER, BOOTSTRAP
from cheatbench_inspect.harness import HarnessPolicy
from cheatbench_inspect.reasoning_replay import anthropic_reasoning_filter

_RESOURCES = Path(__file__).parent / "resources"
_DEST = "/home/user/.nvm/versions/node/v22.23.5/bin/claude"
_STDERR_SCAN_BYTES = 256 * 1024
_START = """
import io, json, sys
sys.stdin = io.StringIO(json.dumps(PAYLOAD))
exec(compile(PAYLOAD.pop("runner"), "<r>", "exec"), {"__name__": "__main__"})
"""


async def _install_cli(version: str) -> None:
    # inspect_swe has no public binary installer; keep its private API here.
    from inspect_swe._claude_code.agentbinary import claude_code_binary_source
    from inspect_swe._util.agentbinary import ensure_agent_binary_installed

    box = sandbox()
    binary = await ensure_agent_binary_installed(claude_code_binary_source(), version, sandbox=box)
    directory = str(PurePosixPath(binary).parent)
    result = await box.exec(
        [
            "sh",
            "-c",
            'mv "$1" "$2" && chown 1000:1000 "$2" && chmod 755 "$2" && ln -sf "$2" /usr/local/bin/claude && rm -rf "$3" && /usr/local/bin/claude --version',
            "sh",
            binary,
            _DEST,
            directory,
        ],
        user="root",
    )
    if not result.success or not result.stdout.strip().startswith(version):
        raise RuntimeError(f"Claude installation failed: {result.stdout} {result.stderr}")


def _payload(state: TaskState, policy: HarnessPolicy, cli_model: str, port: int) -> dict[str, Any]:
    observer = policy.observer
    row = {"id": str(state.sample_id), **state.metadata}
    if observer:
        row.update(observer["row"])
    return {
        "code": _START,
        "runner": (_RESOURCES / "claude/episode.py").read_text(),
        "task": {
            "id": str(state.sample_id),
            "model": cli_model,
            "prompt": state.input_text,
            "row": row,
            "staged": observer["staged"] if observer else state.metadata.get("staged_files", []),
            "tools": {
                "web_search": policy.web_search,
                "web_fetch": policy.web_fetch,
                "mcp_only": policy.mcp_only,
            },
            "max_turns": policy.max_turns,
            "timeout_s": policy.timeout_s,
            "env": {
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
                "ANTHROPIC_API_KEY": "placeholder",
                "DISABLE_AUTOUPDATER": "1",
            },
            "permission_mode": "bypassPermissions",
        },
        "modules": {
            "verifier": (_RESOURCES / "prime_factorization/verifier.py").read_text(),
            "grader": (_RESOURCES / "prime_factorization/sdk_verifier.py").read_text(),
            "observer": observer["code"] if observer else None,
        },
    }


async def run_claude_sdk(
    state: TaskState, policy: HarnessPolicy, *, cli_version: str, cli_model: str
) -> dict[str, Any]:
    """Run the pinned SDK and return its native episode record."""
    await _install_cli(cli_version)
    async with (
        sandbox_agent_bridge(
            model_aliases={
                cli_model: get_model()  # inspect-evals-lint: ignore[IEBP001] -- Harness execution binds the active model.
            },
            filter=anthropic_reasoning_filter,
            forward_generation_config=False,
        ) as bridge
    ):
        result = await sandbox().exec(
            ["python3", "-I", "-c", BOOTSTRAP],
            input=json.dumps(_payload(state, policy, cli_model, bridge.port)),
            timeout=policy.timeout_s + 650,
            user=AGENT_USER,
            env=AGENT_ENV,
        )
    if not result.success:
        raise RuntimeError(result.stderr)
    record = json.loads(result.stdout.splitlines()[-1])
    record["stderr_tail"] = result.stderr[-_STDERR_SCAN_BYTES:]
    return record


def classify_claude_failure(record: dict[str, Any], stderr_tail: str) -> tuple[str | None, bool]:
    """Apply the original trial's failure precedence to the native record."""
    import re

    from cheatbench_inspect.lost_episodes import REASON_LOST, classify, is_harness_timeout

    error = record.get("error")
    if error:
        if isinstance(record.get("install"), dict) and record["install"].get("status") == "failed":
            return error, False
        if error.startswith(
            "max_turns reached before confirmed task completion"
        ) or is_harness_timeout(error):
            return error, False
        reason = classify(error, stderr_tail)
        if (
            reason in (None, "unknown_api_error")
            and re.search(
                r"Command failed with exit code 1|Fatal error in message reader",
                error + "\n" + stderr_tail,
                re.IGNORECASE,
            )
            and classify(record.get("final_text")) == "proxy_5xx"
        ):
            reason = "proxy_5xx"
        return error, REASON_LOST.get(reason or "", False)
    if not (record.get("final_text") or "").strip():
        lines = [line for line in stderr_tail.splitlines() if line.strip()][-20:]
        reason = classify("\n".join(lines))
        if REASON_LOST.get(reason or "", False):
            return f"claude-sdk exited 0 after: {lines[-1].strip()}", True
    return None, False
