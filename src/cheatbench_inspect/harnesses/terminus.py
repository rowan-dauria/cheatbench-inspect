"""Prepare CheatBench's Terminus-2 harness inside the active Inspect sandbox."""

import json
from pathlib import Path

from inspect_ai.agent import AgentState, sandbox_agent_bridge
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from cheatbench_inspect.harnesses.container import (
    AGENT_ENV,
    AGENT_USER,
    BOOTSTRAP,
    SANDBOX_RESOURCES,
)
from cheatbench_inspect.harnesses.harness import HarnessPolicy, HarnessResult

_RESOURCES = Path(__file__).parent / "resources/terminus"
TERMINUS_COMPOSE = SANDBOX_RESOURCES / "terminus-compose.yaml"

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
    result = asyncio.run(runner["episode"](payload["config"], extension["ImageTerminus2"]))
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
    timeout_s: int = 3600,
) -> tuple[list[str], str]:
    """Return the sandbox command and the JSON document to send on its stdin."""
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
        },
        "extension": (_RESOURCES / "image_extension.py").read_text(),
        "runner": (_RESOURCES / "episode.py").read_text(),
    }
    return ["/opt/terminus-runtime/.venv/bin/python", "-I", "-c", BOOTSTRAP], json.dumps(payload)


async def run_terminus_2(state: TaskState, policy: HarnessPolicy) -> HarnessResult:
    """Run the pinned Harbor harness in the sample sandbox."""
    if policy.mcp_only:
        raise ValueError("Terminus-2 does not support mcp_only")
    agent_state = AgentState(messages=state.messages)
    try:
        async with sandbox_agent_bridge(agent_state) as bridge:
            command, stdin = prepare_harbor(
                sample_id=str(state.sample_id),
                prompt=state.input_text,
                model=str(state.model),
                api_base=f"http://127.0.0.1:{bridge.port}/v1",
                mcp_servers=policy.mcp_servers,
                max_turns=policy.max_turns,
                timeout_s=policy.timeout_s,
            )
            result = await sandbox().exec(
                command, input=stdin, timeout=policy.timeout_s + 650, user=AGENT_USER, env=AGENT_ENV
            )
    finally:
        state.messages = agent_state.messages
        if not agent_state.output.empty:
            state.output = agent_state.output
    if not result.success:
        raise RuntimeError(result.stderr)
    data = json.loads(result.stdout)
    return HarnessResult(
        trajectory=data["harbor_trajectory"],
        final_text=data["final_text"],
        error=data["error"],
        artifacts=data.get("harbor_artifacts"),
    )
