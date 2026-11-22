"""Prepare CheatBench's Terminus-2 harness inside the active Inspect sandbox."""

import json
from pathlib import Path
from typing import Any

from inspect_ai.agent import sandbox_agent_bridge
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from cheatbench_inspect.container import AGENT_ENV, AGENT_USER, BOOTSTRAP
from cheatbench_inspect.harness import HarnessPolicy, HarnessResult

_RESOURCES = Path(__file__).parent / "resources/terminus"
TERMINUS_COMPOSE = _RESOURCES.parent / "sandbox/terminus-compose.yaml"

# The episode receives trusted code on stdin. Harbor may print diagnostics during
# imports and inference, so reserve stdout for the completed result.
_START = """
import asyncio, contextlib, json, os, sys
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
payload = PAYLOAD
with contextlib.redirect_stdout(sys.stderr):
    extension, runner = {}, {}
    exec(compile(payload["extension"], "image_extension.py", "exec"), extension)
    exec(compile(payload["runner"], "episode.py", "exec"), runner)
    observer = None
    if payload["observer"]:
        task = {}
        exec(compile(payload["observer"], "observer.py", "exec"), task)
        observer = task["MessageObserver"]
    result = asyncio.run(
        runner["episode"](payload["config"], extension["ImageTerminus2"], observer)
    )
print(json.dumps(result))
"""


def prepare_harbor(
    *,
    sample_id: str,
    prompt: str,
    model: str,
    api_base: str,
    mcp_servers: list[dict[str, str]],
    max_turns: int | None = None,
    observer: dict[str, Any] | None = None,
    timeout_s: int = 3600,
) -> tuple[list[str], str]:
    """Return the sandbox command and the JSON document to send on its stdin.

    `observer` carries the task's live log observer: its `code`, which defines
    `MessageObserver`, and the `row` and `staged` paths it is constructed with.
    """
    payload = {
        "code": _START,
        "config": {
            "id": sample_id,
            "prompt": prompt,
            "model": model,
            "api_base": api_base,
            "mcp_servers": mcp_servers,
            "max_turns": max_turns,
            "timeout_s": timeout_s,
            "observer": {k: v for k, v in observer.items() if k != "code"} if observer else None,
        },
        "extension": (_RESOURCES / "image_extension.py").read_text(),
        "runner": (_RESOURCES / "episode.py").read_text(),
        "observer": observer["code"] if observer else None,
    }
    return ["/opt/terminus-runtime/.venv/bin/python", "-I", "-c", BOOTSTRAP], json.dumps(payload)


async def run_terminus_2(state: TaskState, policy: HarnessPolicy) -> HarnessResult:
    """Run the pinned Harbor harness in the sample sandbox."""
    if policy.mcp_only:
        raise ValueError("Terminus-2 does not support mcp_only")
    async with sandbox_agent_bridge() as bridge:
        command, stdin = prepare_harbor(
            sample_id=str(state.sample_id),
            prompt=state.input_text,
            model=str(state.model),
            api_base=f"http://127.0.0.1:{bridge.port}/v1",
            mcp_servers=policy.mcp_servers,
            max_turns=policy.max_turns,
            observer=policy.observer,
            timeout_s=policy.timeout_s,
        )
        result = await sandbox().exec(
            command, input=stdin, timeout=policy.timeout_s + 650, user=AGENT_USER, env=AGENT_ENV
        )
        if not result.success:
            raise RuntimeError(result.stderr)
        data = json.loads(result.stdout)
        return HarnessResult(
            trajectory=data["harbor_trajectory"],
            final_text=data["final_text"],
            error=data["error"],
            artifacts=data.get("harbor_artifacts"),
            live_logs=data.get("live_logs"),
        )
