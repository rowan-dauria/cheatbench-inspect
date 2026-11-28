"""Run inspect_swe's Kimi Code agent."""

from inspect_ai.agent import AgentState
from inspect_ai.tool import MCPServerConfigHTTP
from inspect_ai.util import sandbox, store
from inspect_swe import kimi_code

from cheatbench_inspect.harnesses.container import AGENT_ENV, AGENT_HOME, AGENT_USER
from cheatbench_inspect.harnesses.harness import HarnessPolicy

_KIMI_HOME = "/opt/kimi"
_TOOL_CONFIG = 'telemetry = false\n\n[tools]\ndisabled = ["FetchURL", "WebSearch"]\n'


async def _prepare_kimi_home() -> None:
    # The sticky directory lets Kimi create session files without replacing root's policy.
    box = sandbox()
    result = await box.exec(
        ["install", "-d", "-o", "root", "-g", "root", "-m", "1777", _KIMI_HOME], user="root"
    )
    if not result.success:
        raise RuntimeError(f"Cannot create Kimi home: {result.stderr}")
    config = f"{_KIMI_HOME}/config.toml"
    await box.write_file(config, _TOOL_CONFIG)
    result = await box.exec(
        [
            "sh",
            "-ec",
            'chown root:root "$1"; chmod 0644 "$1"; ln -sfn "$2" "$3"',
            "sh",
            config,
            f"{AGENT_HOME}/.kimi-code/mcp.json",
            f"{_KIMI_HOME}/mcp.json",
        ],
        user="root",
    )
    if not result.success:
        raise RuntimeError(f"Cannot set up Kimi configuration: {result.stderr}")


async def run_kimi_code(
    state: AgentState, policy: HarnessPolicy, *, cli_version: str, cli_model: str
) -> AgentState:
    """Run one pinned Kimi Code episode inside the sample sandbox."""
    if policy.mcp_only:
        raise ValueError("kimi-code cannot restrict tools to MCP servers")

    await _prepare_kimi_home()
    # inspect-swe 0.2.72 chooses this port and records it when the agent starts.
    # Its config is generated after the bridge starts, and env values are literal.
    port = store().get("kimi_code_model_port", 3100) + 1
    env = {
        **AGENT_ENV,
        **policy.env,
        "KIMI_CODE_HOME": _KIMI_HOME,
        "KIMI_MODEL_NAME": cli_model,
        "KIMI_MODEL_PROVIDER_TYPE": "kimi",
        "KIMI_MODEL_API_KEY": "sk-none",
        "KIMI_MODEL_BASE_URL": f"http://localhost:{port}/v1",
        "KIMI_DISABLE_TELEMETRY": "1",
        "KIMI_CODE_NO_AUTO_UPDATE": "1",
        "NO_COLOR": "1",
    }
    if policy.max_turns:
        env["KIMI_LOOP_MAX_STEPS_PER_TURN"] = str(policy.max_turns)

    # KIMI_MODEL_NAME sets the CLI name. The stock bridge defaults to the active eval model.
    agent = kimi_code(
        version=cli_version,
        max_context_size=262_144,  # Kimi 0.43.0 default for the original env-configured model.
        cwd="/workspace",
        user=AGENT_USER,
        env=env,
        attempts=1,
        retry_refusals=0,
        mcp_servers=[
            MCPServerConfigHTTP(name=server["name"], url=server["url"], type="http")
            for server in policy.mcp_servers
        ],
    )
    return await agent(state)
