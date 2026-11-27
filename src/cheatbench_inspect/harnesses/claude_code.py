"""Run inspect_swe's Claude Code agent."""

from inspect_ai.agent import AgentState
from inspect_ai.tool import MCPServerConfigHTTP
from inspect_swe import claude_code

from cheatbench_inspect.harnesses.container import AGENT_ENV, AGENT_USER
from cheatbench_inspect.harnesses.harness import HarnessPolicy
from cheatbench_inspect.harnesses.reasoning_replay import anthropic_reasoning_filter

_TOOLS = ["Bash", "Read", "Write"]
_DISALLOWED_TOOLS = [
    "Task",
    "TodoWrite",
    "Edit",
    "NotebookEdit",
    "WebSearch",
    "WebFetch",
]


async def run_claude_code(
    state: AgentState, policy: HarnessPolicy, *, cli_version: str, cli_model: str
) -> AgentState:
    """Run one pinned Claude Code episode inside the sample sandbox."""
    agent = claude_code(
        version=cli_version,
        model_config=cli_model,
        cwd="/workspace",
        user=AGENT_USER,
        env={
            **AGENT_ENV,
            **policy.env,
            "DISABLE_AUTOUPDATER": "1",
            # Keep the SDK request metadata and suppress the extra CLI budget reminder.
            "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
            "CLAUDE_CODE_TOTAL_TOKENS_REMINDER": "off",
        },
        permission_mode="bypassPermissions",
        retry_refusals=0,
        retry_uncaught_errors=0,
        filter=anthropic_reasoning_filter,
        tools=_TOOLS,
        disallowed_tools=_DISALLOWED_TOOLS,
        strict_mcp_config=True,
        replace_system_prompt="",
        mcp_servers=[
            MCPServerConfigHTTP(name=server["name"], url=server["url"], type="http")
            for server in policy.mcp_servers
        ],
    )
    return await agent(state)
