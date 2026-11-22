"""Host-side SDK payload checks."""

from types import SimpleNamespace
from typing import cast

from inspect_ai.solver import TaskState

from cheatbench_inspect.claude_sdk import _payload
from cheatbench_inspect.harness import HarnessPolicy


def test_payload_preserves_original_model_and_policy():
    state = SimpleNamespace(
        sample_id="q1",
        input_text="factor 15",
        metadata={"number": 15, "staged_files": ["/workspace/a"]},
    )
    policy = HarnessPolicy(mcp_servers=[], max_turns=3, timeout_s=30, observer=None)
    payload = _payload(cast(TaskState, state), policy, "claude-sonnet-5", 1234)
    task = payload["task"]
    assert task["model"] == "claude-sonnet-5"
    assert task["row"]["number"] == 15
    assert task["staged"] == ["/workspace/a"]
    assert task["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1234"
    assert task["max_turns"] == 3
    assert payload["modules"]["grader"]


async def test_launcher_sends_payload_to_sandbox(monkeypatch):
    import json
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock

    import cheatbench_inspect.claude_sdk as launcher

    state = cast(
        TaskState, SimpleNamespace(sample_id="15", input_text="factor", metadata={"number": 15})
    )
    policy = HarnessPolicy(mcp_servers=[], max_turns=4, timeout_s=60, observer=None)
    box = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                success=True, stdout='{"final_text":"done"}\n', stderr="warning"
            )
        )
    )
    install = AsyncMock()
    active = object()
    captured = {}

    @asynccontextmanager
    async def bridge(**kwargs):
        captured.update(kwargs)
        yield SimpleNamespace(port=1234)

    monkeypatch.setattr(launcher, "_install_cli", install)
    monkeypatch.setattr(launcher, "sandbox", lambda: box)
    monkeypatch.setattr(launcher, "get_model", lambda: active)
    monkeypatch.setattr(launcher, "sandbox_agent_bridge", bridge)
    result = await launcher.run_claude_sdk(
        state, policy, cli_version="2.1.270", cli_model="original"
    )
    install.assert_awaited_once_with("2.1.270")
    assert captured["model_aliases"] == {"original": active}
    assert captured["forward_generation_config"] is False
    args = box.exec.await_args.kwargs
    assert json.loads(args["input"])["task"]["model"] == "original"
    assert args["user"] == "user"
    assert args["timeout"] == 710
    assert result == {"final_text": "done", "stderr_tail": "warning"}
