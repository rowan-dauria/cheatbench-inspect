"""Unit tests for generic MCP-only policy enforcement across all four harnesses."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.model import (
    ChatMessageAssistant,
    ContentReasoning,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import MCPServerConfigHTTP, ToolInfo, ToolParams

import cheatbench_inspect.harnesses.claude_code as claude
import cheatbench_inspect.harnesses.codex as codex
import cheatbench_inspect.harnesses.terminus as terminus
from cheatbench_inspect.harnesses.harness import HarnessPolicy, mcp_only_tools


def test_mcp_only_tools_validation() -> None:
    policy = HarnessPolicy(
        [{"name": "board", "url": "http://127.0.0.1:1234/mcp"}],
        mcp_only=True,
        mcp_tools={"board": ("move", "status")},
    )
    assert mcp_only_tools(policy) == {"board": ("move", "status")}

    with pytest.raises(ValueError, match="at least one MCP server"):
        mcp_only_tools(HarnessPolicy([], mcp_only=True))
    with pytest.raises(ValueError, match="nonempty name and url"):
        mcp_only_tools(HarnessPolicy([{"name": "", "url": "http://x"}], mcp_only=True))
    with pytest.raises(ValueError, match="Duplicate MCP server"):
        mcp_only_tools(
            HarnessPolicy(
                [{"name": "a", "url": "http://1"}, {"name": "a", "url": "http://2"}],
                mcp_only=True,
                mcp_tools={"a": ("x",)},
            )
        )
    with pytest.raises(ValueError, match="unknown server"):
        mcp_only_tools(
            HarnessPolicy(
                [{"name": "a", "url": "http://1"}],
                mcp_only=True,
                mcp_tools={"a": ("x",), "other": ("y",)},
            )
        )
    with pytest.raises(ValueError, match="nonempty allowed tools"):
        mcp_only_tools(HarnessPolicy([{"name": "a", "url": "http://1"}], mcp_only=True))
    with pytest.raises(ValueError, match="Duplicate allowed tool names"):
        mcp_only_tools(
            HarnessPolicy(
                [{"name": "a", "url": "http://1"}],
                mcp_only=True,
                mcp_tools={"a": ("x", "x")},
            )
        )


async def test_claude_mcp_only_launcher() -> None:
    state = AgentState(messages=[])
    policy = HarnessPolicy(
        [{"name": "board", "url": "http://127.0.0.1:9999/mcp"}],
        mcp_only=True,
        mcp_tools={"board": ("move", "status")},
    )
    agent = AsyncMock(return_value=state)
    with patch.object(claude, "claude_code", return_value=agent) as factory:
        assert (
            await claude.run_claude_code(
                state, policy, cli_version="2.1.270", cli_model="claude-sonnet-5"
            )
            is state
        )
    kwargs = factory.call_args.kwargs
    assert kwargs["tools"] == []
    assert kwargs["strict_mcp_config"] is True
    assert len(kwargs["disallowed_tools"]) == 14
    assert {"Bash", "Read", "Write", "Agent", "Glob", "Grep", "Skill", "ToolSearch"} <= set(
        kwargs["disallowed_tools"]
    )
    assert kwargs["mcp_servers"] == [
        MCPServerConfigHTTP(name="board", url="http://127.0.0.1:9999/mcp", type="http")
    ]


async def test_codex_mcp_only_overrides_and_filter() -> None:
    policy = HarnessPolicy(
        [{"name": "board", "url": "http://127.0.0.1:9999/mcp"}],
        mcp_only=True,
        mcp_tools={"board": ("move", "status")},
        developer_instructions="Custom MCP instructions.",
    )
    overrides = codex._config_overrides(policy, GenerateConfig(reasoning_effort="high"))
    assert overrides["features.shell_tool"] == "false"
    assert overrides["features.unified_exec"] == "false"
    assert overrides["features.code_mode.enabled"] == "false"
    assert overrides["features.code_mode.direct_only_tool_namespaces"] == '["mcp__board"]'
    assert overrides["mcp_servers.board.required"] == "true"
    assert overrides["mcp_servers.board.enabled_tools"] == '["move", "status"]'
    assert overrides["developer_instructions"] == '"Custom MCP instructions."'

    # Filter keeps only allowlisted namespaced or flat MCP tools and preserves reasoning replay.
    seen: dict[str, Any] = {}

    class FakeModel:
        async def generate(
            self, messages: Any, tools: Any, tool_choice: Any, config: Any
        ) -> ModelOutput:
            seen["messages"] = messages
            seen["tools"] = [t.name for t in tools]
            return ModelOutput.from_content("mockllm/model", "ok")

    tools = [
        ToolInfo(
            name="exec",
            description="shell",
            parameters=ToolParams(),
            options={"__responses_namespace__": ("functions", "functions")},
        ),
        ToolInfo(
            name="sleep",
            description="clock",
            parameters=ToolParams(),
            options={"__responses_namespace__": ("clock", "clock")},
        ),
        ToolInfo(
            name="move",
            description="board move",
            parameters=ToolParams(),
            options={"__responses_namespace__": ("mcp__board", "mcp__board")},
        ),
        ToolInfo(
            name="unlisted",
            description="extra",
            parameters=ToolParams(),
            options={"__responses_namespace__": ("mcp__board", "mcp__board")},
        ),
        ToolInfo(
            name="mcp__board__status",
            description="flat status",
            parameters=ToolParams(),
        ),
    ]
    filt = codex._mcp_only_responses_filter({"board": ("move", "status")})
    msg = ChatMessageAssistant(
        content=[ContentReasoning(reasoning="enc", summary="plan", redacted=True)]
    )
    await filt(FakeModel(), [msg], tools, "auto", GenerateConfig())  # type: ignore[arg-type]
    assert seen["tools"] == ["move", "mcp__board__status"]
    part = seen["messages"][0].content[0]
    assert part.redacted is False
    assert part.summary == "plan"
    assert part.internal["reasoning_encrypted_content"] == "enc"

    # run_codex wires sandbox_mode="read-only", goals=False, and the MCP-only filter.
    state = AgentState(messages=[])
    agent = AsyncMock(return_value=state)
    fake_or = get_model(
        "openrouter/openai/gpt-6-astra",
        base_url="http://127.0.0.1:1/api/v1",
        api_key="dummy",
        responses_api=True,
    )
    with (
        patch.object(codex, "get_model", return_value=fake_or),
        patch.object(codex, "codex_cli", return_value=agent) as factory,
    ):
        await codex.run_codex(state, policy, cli_version="0.154.0", cli_model="gpt-6-astra")
    assert factory.call_args.kwargs["sandbox_mode"] == "read-only"
    assert factory.call_args.kwargs["goals"] is False
    assert factory.call_args.kwargs["filter"] is not codex.responses_reasoning_filter


def test_terminus_prepare_harbor_mcp_only() -> None:
    cmd, raw = terminus.prepare_harbor(
        sample_id="s1",
        prompt="Play.",
        model="mockllm/model",
        api_base="http://127.0.0.1:1234/v1",
        mcp_servers=[
            {"name": "board", "transport": "streamable-http", "url": "http://127.0.0.1:9/mcp"}
        ],
        max_turns=10,
        timeout_s=60,
        mcp_only=True,
        mcp_tools={"board": ("move", "status")},
    )
    assert cmd[0] == "/opt/terminus-runtime/.venv/bin/python"
    payload = json.loads(raw)
    assert payload["config"]["mcp_only"] is True
    assert payload["config"]["mcp_tools"] == {"board": ["move", "status"]}
    assert "MCPImageTerminus2" in payload["mcp_extension"]


def test_terminus_mcp_only_parser_edge_cases() -> None:
    from types import SimpleNamespace

    class FakeImageParser:
        def __init__(self) -> None:
            self.image_paths: list[str] = []

        def _extract_json_content(self, text: str) -> tuple[str, str]:
            start = text.index("{")
            end = text.rindex("}") + 1
            return text[start:end], ""

        def _try_parse_response(self, text: str) -> SimpleNamespace:
            raw, _ = self._extract_json_content(text)
            doc = json.loads(raw)
            return SimpleNamespace(error="", is_task_complete=bool(doc.get("task_complete")))

    class FakeImageTerminus2:
        pass

    def fake_split(text: str) -> tuple[str, str | None]:
        if text.startswith("<think>") and "</think>" in text:
            head, rest = text.split("</think>", 1)
            return rest.lstrip(), head[len("<think>") :]
        return text, None

    ns: dict[str, Any] = {
        "ImageParser": FakeImageParser,
        "ImageTerminus2": FakeImageTerminus2,
        "ToolCall": object,
        "Observation": object,
        "ObservationResult": object,
        "_split_bridge_reasoning": fake_split,
    }
    exec((terminus._RESOURCES / "mcp_extension.py").read_text(encoding="utf-8"), ns)
    parser = ns["MCPOnlyParser"]()

    # Braces inside <think> block do not break JSON extraction.
    ok = parser._try_parse_response(
        '<think>Consider {"x": 1} before acting.</think>\n'
        '{"analysis": "a", "plan": "p", "commands": [], '
        '"mcp_calls": [{"name": "game_status", "arguments": {}}], "task_complete": false}'
    )
    assert not ok.error
    assert parser.mcp_calls == [{"name": "game_status", "arguments": {}}]

    # Shell commands, view_images, malformed calls, and calls+task_complete are rejected.
    for payload, expected in (
        ('{"commands": [{"keystrokes": "ls\\n"}]}', "MCP tools only"),
        ('{"commands": [], "view_images": ["/workspace/a.png"]}', "MCP tools only"),
        ('{"commands": [], "mcp_calls": [{"name": 123}]}', "name and arguments"),
        (
            '{"commands": [], "mcp_calls": [{"name": "game_status", "arguments": {}}], "task_complete": true}',
            "before marking task_complete",
        ),
    ):
        res = parser._try_parse_response(payload)
        assert expected in res.error
        assert parser.mcp_calls == []
