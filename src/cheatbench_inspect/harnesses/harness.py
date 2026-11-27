"""Shared interface for agent harnesses."""

from dataclasses import dataclass, field
from typing import Any

from inspect_ai.agent import AgentState
from inspect_ai.model import ModelOutput
from inspect_ai.solver import TaskState

from cheatbench_inspect.harnesses.lost_episodes import is_lost_episode


@dataclass(frozen=True)
class HarnessPolicy:
    mcp_servers: list[dict[str, str]]
    max_turns: int | None = None
    timeout_s: int = 3600
    mcp_only: bool = False
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class HarnessResult:
    trajectory: dict[str, Any] | None
    final_text: str
    error: str | None
    artifacts: Any


async def _terminus_2(state: TaskState, policy: HarnessPolicy) -> HarnessResult:
    from cheatbench_inspect.harnesses.terminus import run_terminus_2

    return await run_terminus_2(state, policy)


async def _claude_code(
    state: AgentState, policy: HarnessPolicy, options: dict[str, str]
) -> AgentState:
    from cheatbench_inspect.harnesses.claude_code import run_claude_code

    return await run_claude_code(
        state, policy, cli_version=options["cli_version"], cli_model=options["cli_model"]
    )


async def _codex(state: AgentState, policy: HarnessPolicy, options: dict[str, str]) -> AgentState:
    from cheatbench_inspect.harnesses.codex import run_codex

    return await run_codex(
        state, policy, cli_version=options["cli_version"], cli_model=options["cli_model"]
    )


async def _gemini_cli(
    state: AgentState, policy: HarnessPolicy, options: dict[str, str]
) -> AgentState:
    from cheatbench_inspect.harnesses.gemini_cli import run_gemini_cli

    return await run_gemini_cli(
        state, policy, cli_version=options["cli_version"], cli_model=options["cli_model"]
    )


async def _kimi_code(
    state: AgentState, policy: HarnessPolicy, options: dict[str, str]
) -> AgentState:
    from cheatbench_inspect.harnesses.kimi_code import run_kimi_code

    return await run_kimi_code(
        state, policy, cli_version=options["cli_version"], cli_model=options["cli_model"]
    )


_HARNESSES = {
    "terminus-2": _terminus_2,
    "claude-code": _claude_code,
    "codex": _codex,
    "gemini-cli": _gemini_cli,
    "kimi-code": _kimi_code,
}
CLI_HARNESSES = {"claude-code", "codex", "gemini-cli", "kimi-code"}
_CLI_OPTIONS = {"cli_version", "cli_model"}


def validate_harness(name: str) -> None:
    if name not in _HARNESSES:
        raise ValueError(f"Unknown harness {name!r}; known harnesses: {', '.join(_HARNESSES)}")


async def run_harness(
    name: str, state: TaskState, policy: HarnessPolicy, options: dict[str, str] | None = None
) -> HarnessResult | AgentState:
    validate_harness(name)
    if name in CLI_HARNESSES:
        if not options or not options.get("cli_version") or not options.get("cli_model"):
            raise ValueError(f"{name} requires cli_version and cli_model")
        unknown = sorted(set(options) - _CLI_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown options for {name}: {', '.join(unknown)}")
        agent_state = AgentState(messages=state.messages)
        try:
            launch = _HARNESSES[name]
            agent_state = await launch(agent_state, policy, options)
            return agent_state
        finally:
            state.messages = agent_state.messages
            if not agent_state.output.empty:
                state.output = agent_state.output
    if options:
        raise ValueError("terminus-2 does not accept harness options")
    return await _terminus_2(state, policy)


def apply_harness_result(state: TaskState, result: HarnessResult) -> None:
    state.output = ModelOutput.from_content(model=str(state.model), content=result.final_text)
    state.store.set("trajectory", result.trajectory)
    state.store.set("harness_artifacts", result.artifacts)
    state.store.set("harness_error", result.error)
    if result.trajectory is None or (result.error and is_lost_episode(result.error)):
        raise RuntimeError(result.error or "Harness produced no trajectory")
