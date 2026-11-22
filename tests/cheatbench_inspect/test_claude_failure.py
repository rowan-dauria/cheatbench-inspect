"""Failure precedence for the Claude SDK harness."""

import pytest

from cheatbench_inspect.claude_sdk import classify_claude_failure


@pytest.mark.parametrize(
    ("record", "stderr", "lost"),
    [
        ({"error": "max_turns reached before confirmed task completion"}, "502 Bad Gateway", False),
        ({"error": "timeout after 30s"}, "502 Bad Gateway", False),
        ({"error": "502 Bad Gateway"}, "", True),
        ({"error": "Command failed with exit code 1", "final_text": "502 Bad Gateway"}, "", True),
        ({"final_text": "A 502 Bad Gateway occurred earlier"}, "502 Bad Gateway", False),
        ({"final_text": "success"}, "502 Bad Gateway", False),
        ({"final_text": ""}, "502 Bad Gateway", True),
    ],
)
def test_failure_precedence(record: dict, stderr: str, lost: bool) -> None:
    assert classify_claude_failure(record, stderr)[1] is lost


@pytest.mark.parametrize(
    ("harness", "version", "model"),
    [
        ("claude-sdk", None, "model"),
        ("claude-sdk", "1", None),
        ("terminus-2", "1", None),
        ("terminus-2", None, "model"),
    ],
)
def test_task_rejects_invalid_harness_options(
    harness: str, version: str | None, model: str | None
) -> None:
    from cheatbench_inspect.tasks import cheatbench_prime_factorization

    with pytest.raises(ValueError, match="harness_version"):
        cheatbench_prime_factorization(
            harness=harness, harness_version=version, harness_model=model
        )


async def test_no_assistant_message_is_sample_error(monkeypatch) -> None:
    from inspect_ai.model import ModelName
    from inspect_ai.solver import TaskState

    import cheatbench_inspect.claude_sdk as sdk
    from cheatbench_inspect.harness import HarnessPolicy, apply_harness_result, run_harness

    async def run(*args, **kwargs):
        return {"messages": [], "final_text": "", "error": None}

    monkeypatch.setattr(sdk, "run_claude_sdk", run)
    state = TaskState(
        model=ModelName("mockllm/model"), sample_id="1", epoch=1, input="test", messages=[]
    )
    policy = HarnessPolicy(mcp_servers=[], max_turns=1, timeout_s=10, observer=None)
    result = await run_harness(
        "claude-sdk", state, policy, {"cli_version": "1", "cli_model": "model"}
    )
    assert result.trajectory is None
    with pytest.raises(RuntimeError, match="no trajectory"):
        apply_harness_result(state, result)
