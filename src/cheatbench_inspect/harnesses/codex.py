# Codex feature settings adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/codex.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.
"""Run inspect_swe's Codex CLI agent."""

import json

from inspect_ai.agent import AgentState
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    GenerateFilter,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.model._generate_config import (
    active_generate_config,  # inspect-evals-lint: ignore[private_api_imports] -- No public accessor for the active eval config.
)
from inspect_ai.tool import MCPServerConfigHTTP, ToolChoice, ToolInfo
from inspect_swe import codex_cli

from cheatbench_inspect.harnesses.container import AGENT_ENV, AGENT_USER
from cheatbench_inspect.harnesses.harness import HarnessPolicy, mcp_only_tools
from cheatbench_inspect.harnesses.reasoning_replay import responses_reasoning_filter

# No marketplace sync or optional plugin tools in the episode.
_DISABLED_FEATURES = ("image_generation", "browser_use", "computer_use", "apps", "plugins")
_MCP_ONLY_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "shell_snapshot",
    "multi_agent",
    "code_mode_host",
    "remote_plugin",
    "view_image",
    "workspace_dependencies",
    "goals",
    "tool_suggest",
    "enable_request_compression",
)


def _config_overrides(policy: HarnessPolicy, config: GenerateConfig) -> dict[str, str]:
    overrides = {
        "analytics.enabled": "false",
        "check_for_update_on_startup": "false",
        **{f"features.{name}": "false" for name in _DISABLED_FEATURES},
    }
    effort = config.reasoning_effort if config.reasoning_effort is not None else "high"
    if effort:
        overrides["model_reasoning_effort"] = json.dumps(effort)
    if config.reasoning_summary and config.reasoning_summary != "none":
        if config.reasoning_summary not in ("auto", "concise", "detailed"):
            raise ValueError(f"Unsupported Codex reasoning_summary={config.reasoning_summary!r}")
        overrides["model_reasoning_summary"] = json.dumps(config.reasoning_summary)
    if not policy.mcp_only:
        overrides["features.unified_exec"] = "true"
        return overrides

    allowed = mcp_only_tools(policy)
    server_names = list(allowed)
    overrides.update({f"features.{name}": "false" for name in _MCP_ONLY_DISABLED_FEATURES})
    overrides.update(
        {
            "features.code_mode.direct_only_tool_namespaces": json.dumps(
                [f"mcp__{name}" for name in server_names]
            ),
            "features.code_mode.enabled": "false",
            "agents.enabled": "false",
        }
    )
    for name, tools in allowed.items():
        overrides.update(
            {
                f"mcp_servers.{name}.required": "true",
                f"mcp_servers.{name}.tool_timeout_sec": "900",
                f"mcp_servers.{name}.enabled_tools": json.dumps(list(tools)),
                f"mcp_servers.{name}.default_tools_approval_mode": '"approve"',
            }
        )
    instructions = policy.developer_instructions or (
        f"The local filesystem sandbox is read-only. The listed {', '.join(server_names)} MCP "
        "tools operate on task-managed state through the server. Follow the task instructions "
        "when deciding how to use the tools."
    )
    overrides["developer_instructions"] = json.dumps(instructions)
    return overrides


def _is_allowed_mcp_tool(tool: ToolInfo, allowed_by_ns: dict[str, set[str]]) -> bool:
    ns = (tool.options or {}).get("__responses_namespace__")
    if (
        isinstance(ns, (tuple, list))
        and len(ns) == 2
        and isinstance(ns[0], str)
        and ns[0] in allowed_by_ns
        and tool.name in allowed_by_ns[ns[0]]
    ):
        return True
    for ns_name, names in allowed_by_ns.items():
        prefix = f"{ns_name}__"
        if tool.name.startswith(prefix) and tool.name[len(prefix) :] in names:
            return True
    return False


def _mcp_only_responses_filter(allowed: dict[str, tuple[str, ...]]) -> GenerateFilter:
    """Keep only allowlisted MCP server tools and replay Responses reasoning."""
    allowed_by_ns = {f"mcp__{server}": set(tools) for server, tools in allowed.items()}

    async def _filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput:
        filtered = [tool for tool in tools if _is_allowed_mcp_tool(tool, allowed_by_ns)]
        return await responses_reasoning_filter(model, messages, filtered, tool_choice, config)

    return _filter


async def run_codex(
    state: AgentState, policy: HarnessPolicy, *, cli_version: str, cli_model: str
) -> AgentState:
    """Run one pinned Codex CLI episode inside the sample sandbox."""
    model = get_model()  # inspect-evals-lint: ignore[IEBP001] -- Binds the active model.
    validate_responses_route(model)
    config = model.config.merge(active_generate_config())
    overrides = _config_overrides(policy, config)
    request_filter: GenerateFilter = (
        _mcp_only_responses_filter(mcp_only_tools(policy))
        if policy.mcp_only
        else responses_reasoning_filter
    )
    slug = cli_model.rsplit("/", 1)[-1]
    agent = codex_cli(
        version=cli_version,
        model_config=slug,
        cwd="/workspace",
        user=AGENT_USER,
        home_dir="/tmp/codex-home",
        env={**AGENT_ENV, **policy.env},
        web_search="disabled",
        goals=not policy.mcp_only,
        attempts=1,
        retry_refusals=0,
        sandbox_mode="read-only" if policy.mcp_only else "danger-full-access",
        config_overrides=overrides,
        filter=request_filter,
        mcp_servers=[
            MCPServerConfigHTTP(name=server["name"], url=server["url"], type="http")
            for server in policy.mcp_servers
        ],
    )
    return await agent(state)


def validate_responses_route(model: Model) -> None:
    """Fail loudly if the active Inspect model is not routed over the Responses API."""
    api = model.api
    uses_responses = getattr(api, "uses_responses_api", None)
    if callable(uses_responses):
        is_responses = bool(uses_responses([], model.config))
    else:
        is_responses = bool(getattr(api, "responses_api", False))
    if not is_responses:
        raise ValueError(
            "Codex harness requires a model served over the OpenAI Responses API, "
            f"but active model {model!s} ({type(api).__name__}) is not using the "
            "Responses API. For OpenRouter models, pass model arg `responses_api=true`."
        )
