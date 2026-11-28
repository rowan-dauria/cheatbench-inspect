"""Run inspect_swe's Gemini CLI agent."""

import json

from inspect_ai.agent import AgentState
from inspect_ai.tool import MCPServerConfigHTTP
from inspect_ai.util import sandbox
from inspect_swe import gemini_cli

from cheatbench_inspect.harnesses.container import user_env
from cheatbench_inspect.harnesses.harness import HarnessPolicy

_SYSTEM_SETTINGS = "/etc/gemini-cli/system-settings.json"


async def run_gemini_cli(
    state: AgentState, policy: HarnessPolicy, *, cli_version: str, cli_model: str
) -> AgentState:
    """Run one pinned Gemini CLI episode inside the sample sandbox."""
    if policy.mcp_only:
        raise ValueError("gemini-cli cannot restrict tools to MCP servers")
    box = sandbox()
    result = await box.exec(["mkdir", "-p", "/etc/gemini-cli"], user="root")
    if not result.success:
        raise RuntimeError(f"Cannot create Gemini settings directory: {result.stderr}")
    settings: dict[str, object] = {
        "general": {"enableAutoUpdate": False, "enableAutoUpdateNotification": False},
        "advanced": {"autoConfigureMemory": False},
    }
    if not policy.allow_web_tools:
        settings = {"tools": {"exclude": ["google_web_search", "web_fetch"]}, **settings}
    await box.write_file(_SYSTEM_SETTINGS, json.dumps(settings))
    result = await box.exec(["chown", "root:root", _SYSTEM_SETTINGS], user="root")
    if not result.success:
        raise RuntimeError(f"Cannot own Gemini settings: {result.stderr}")
    result = await box.exec(["chmod", "0644", _SYSTEM_SETTINGS], user="root")
    if not result.success:
        raise RuntimeError(f"Cannot protect Gemini settings: {result.stderr}")
    agent = gemini_cli(
        version=cli_version,
        gemini_model=cli_model.rsplit("/", 1)[-1],
        cwd=policy.cwd,
        user=policy.user,
        env={
            **user_env(policy.user),
            **policy.env,
            "NO_COLOR": "1",
            "GEMINI_CLI_SYSTEM_SETTINGS_PATH": _SYSTEM_SETTINGS,
        },
        web_search=policy.allow_web_tools,
        attempts=1,
        retry_refusals=0,
        mcp_servers=[
            MCPServerConfigHTTP(name=server["name"], url=server["url"], type="http")
            for server in policy.mcp_servers
        ],
    )
    return await agent(state)
