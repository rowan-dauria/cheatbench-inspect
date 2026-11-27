"""Gemini CLI launcher and dispatch."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState
from inspect_ai.tool import MCPServerConfigHTTP

import cheatbench_inspect.harnesses.gemini_cli as gemini
from cheatbench_inspect.harnesses.harness import HarnessPolicy, run_harness, validate_harness


async def test_launcher() -> None:
    state = AgentState(messages=[])
    policy = HarnessPolicy([{"name": "verifier", "url": "http://localhost:123/mcp"}])
    agent = AsyncMock(return_value=state)
    box = MagicMock()
    box.exec = AsyncMock(return_value=MagicMock(success=True))
    box.write_file = AsyncMock()
    with (
        patch.object(gemini, "sandbox", return_value=box),
        patch.object(gemini, "gemini_cli", return_value=agent) as factory,
    ):
        assert (
            await gemini.run_gemini_cli(
                state, policy, cli_version="0.59.0", cli_model="google/gemini-3.8-flash"
            )
            is state
        )
    factory.assert_called_once_with(
        version="0.59.0",
        gemini_model="gemini-3.8-flash",
        cwd="/workspace",
        user="user",
        env={
            "HOME": "/home/user",
            "USER": "user",
            "LOGNAME": "user",
            "NO_COLOR": "1",
            "GEMINI_CLI_SYSTEM_SETTINGS_PATH": "/etc/gemini-cli/system-settings.json",
        },
        web_search=False,
        attempts=1,
        retry_refusals=0,
        mcp_servers=[
            MCPServerConfigHTTP(name="verifier", url="http://localhost:123/mcp", type="http")
        ],
    )
    agent.assert_awaited_once_with(state)
    path, content = box.write_file.await_args.args
    assert path == "/etc/gemini-cli/system-settings.json"
    assert json.loads(content) == {
        "tools": {"exclude": ["google_web_search", "web_fetch"]},
        "general": {"enableAutoUpdate": False, "enableAutoUpdateNotification": False},
        "advanced": {"autoConfigureMemory": False},
    }


async def test_mcp_only_writes_core_tools_and_server_filter() -> None:
    state = AgentState(messages=[])
    policy = HarnessPolicy(
        [{"name": "board", "url": "http://localhost:123/mcp"}],
        mcp_only=True,
        mcp_tools={"board": ("play_move", "read_file", "write_file")},
    )
    agent = AsyncMock(return_value=state)
    box = MagicMock()
    box.exec = AsyncMock(return_value=MagicMock(success=True))
    box.write_file = AsyncMock()
    with (
        patch.object(gemini, "sandbox", return_value=box),
        patch.object(gemini, "gemini_cli", return_value=agent) as factory,
    ):
        assert (
            await gemini.run_gemini_cli(
                state, policy, cli_version="0.59.0", cli_model="google/gemini-3.8-flash"
            )
            is state
        )
    assert factory.call_args.kwargs["mcp_servers"] == [
        MCPServerConfigHTTP(
            name="board",
            url="http://localhost:123/mcp",
            type="http",
            tools=["play_move", "read_file", "write_file"],
        )
    ]
    path, content = box.write_file.await_args.args
    assert path == "/etc/gemini-cli/system-settings.json"
    assert json.loads(content) == {
        "tools": {
            "exclude": ["google_web_search", "web_fetch"],
            "core": ["mcp_board_play_move", "mcp_board_read_file", "mcp_board_write_file"],
        },
        "general": {"enableAutoUpdate": False, "enableAutoUpdateNotification": False},
        "advanced": {"autoConfigureMemory": False},
    }


async def test_mcp_only_without_servers_rejected() -> None:
    with pytest.raises(ValueError, match="MCP-only policy requires at least one MCP server"):
        await gemini.run_gemini_cli(
            AgentState(messages=[]),
            HarnessPolicy([], mcp_only=True),
            cli_version="0.59.0",
            cli_model="gemini",
        )


async def test_dispatch_and_validation() -> None:
    validate_harness("gemini-cli")
    state = TaskState(
        model=ModelName("mockllm/model"), sample_id="1", epoch=1, input="test", messages=[]
    )
    policy = HarnessPolicy([])
    result = AgentState(messages=[])
    with patch.object(gemini, "run_gemini_cli", AsyncMock(return_value=result)) as launch:
        assert (
            await run_harness(
                "gemini-cli", state, policy, {"cli_version": "0.59.0", "cli_model": "gemini"}
            )
            is result
        )
        launch.assert_awaited_once()
    for options in (None, {"cli_version": "0.59.0"}, {"cli_model": "gemini"}):
        with pytest.raises(ValueError, match="requires cli_version and cli_model"):
            await run_harness("gemini-cli", state, policy, options)
    with pytest.raises(ValueError, match="Unknown options"):
        await run_harness(
            "gemini-cli",
            state,
            policy,
            {"cli_version": "0.59.0", "cli_model": "gemini", "extra": "x"},
        )


@pytest.mark.parametrize(
    "options", [{}, {"harness_version": "0.59.0"}, {"harness_model": "gemini"}]
)
def test_task_requires_gemini_options(options: dict[str, Any]) -> None:
    from cheatbench_inspect.tasks import cheatbench_prime_factorization

    with pytest.raises(ValueError, match="gemini-cli requires harness_version and harness_model"):
        cheatbench_prime_factorization(harness="gemini-cli", **options)
