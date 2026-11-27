"""Claude Code launcher and dispatch."""

from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState
from inspect_ai.tool import MCPServerConfigHTTP

import cheatbench_inspect.harnesses.claude_code as claude
from cheatbench_inspect.harnesses.harness import HarnessPolicy, run_harness, validate_harness
from cheatbench_inspect.harnesses.reasoning_replay import anthropic_reasoning_filter


async def test_launcher() -> None:
    state = AgentState(messages=[])
    policy = HarnessPolicy(
        [{"name": "verifier", "url": "http://localhost:123/mcp"}],
        env={"NODE_EXTRA_CA_CERTS": "/etc/ssl/extra.pem"},
    )
    agent = AsyncMock(return_value=state)
    with patch.object(claude, "claude_code", return_value=agent) as factory:
        assert (
            await claude.run_claude_code(
                state, policy, cli_version="2.1.270", cli_model="claude-sonnet-5"
            )
            is state
        )
    factory.assert_called_once_with(
        version="2.1.270",
        model_config="claude-sonnet-5",
        cwd="/workspace",
        user="user",
        env={
            "HOME": "/home/user",
            "USER": "user",
            "LOGNAME": "user",
            "NODE_EXTRA_CA_CERTS": "/etc/ssl/extra.pem",
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
            "CLAUDE_CODE_TOTAL_TOKENS_REMINDER": "off",
        },
        permission_mode="bypassPermissions",
        retry_refusals=0,
        retry_uncaught_errors=0,
        filter=anthropic_reasoning_filter,
        tools=["Bash", "Read", "Write"],
        disallowed_tools=[
            "Task",
            "TodoWrite",
            "Edit",
            "NotebookEdit",
            "WebSearch",
            "WebFetch",
        ],
        strict_mcp_config=True,
        replace_system_prompt="",
        mcp_servers=[
            MCPServerConfigHTTP(name="verifier", url="http://localhost:123/mcp", type="http")
        ],
    )
    agent.assert_awaited_once_with(state)


async def test_launcher_without_mcp_servers() -> None:
    state = AgentState(messages=[])
    agent = AsyncMock(return_value=state)
    with patch.object(claude, "claude_code", return_value=agent) as factory:
        assert (
            await claude.run_claude_code(
                state, HarnessPolicy([]), cli_version="2.1.281", cli_model="claude-fable-5-1"
            )
            is state
        )
    assert factory.call_args.kwargs["mcp_servers"] == []
    assert factory.call_args.kwargs["tools"] == ["Bash", "Read", "Write"]
    assert factory.call_args.kwargs["strict_mcp_config"] is True


async def test_dispatch_and_validation() -> None:
    validate_harness("claude-code")
    state = TaskState(
        model=ModelName("mockllm/model"), sample_id="1", epoch=1, input="test", messages=[]
    )
    policy = HarnessPolicy([])
    result = AgentState(messages=[])
    with patch.object(claude, "run_claude_code", AsyncMock(return_value=result)) as launch:
        assert (
            await run_harness(
                "claude-code",
                state,
                policy,
                {"cli_version": "2.1.270", "cli_model": "claude-sonnet-5"},
            )
            is result
        )
        launch.assert_awaited_once()
    for options in (None, {"cli_version": "2.1.270"}, {"cli_model": "claude-sonnet-5"}):
        with pytest.raises(ValueError, match="requires cli_version and cli_model"):
            await run_harness("claude-code", state, policy, options)
