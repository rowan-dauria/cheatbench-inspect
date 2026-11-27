"""Stage the authenticated leaderboard and run one workshop attempt."""

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from secrets import token_hex
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.sandbox_task import sandbox_service_endpoint, sandbox_solver
from cheatbench_inspect.svgs.dataset import CONFIG_PATH, RESOURCES


@asynccontextmanager
async def leaderboard_endpoint(rubric: str, captured: dict[str, Any]) -> AsyncGenerator[str, None]:
    token = token_hex(24)
    config = {
        "code": (RESOURCES / "leaderboard.py").read_text(encoding="utf-8")
        + '\nserve(PAYLOAD["rubric"], PAYLOAD["token"], PAYLOAD["worker"])',
        "worker": (RESOURCES / "render_worker.txt").read_text(encoding="utf-8"),
        "rubric": rubric,
        "token": token,
    }
    async with sandbox_service_endpoint(config, user="root", grader_state=captured) as url:
        # Inspect stages the client before this solver hands ownership to the agent.
        agent_config = {"round": "final", "rubric": rubric, "endpoint": url, "token": token}
        await sandbox().write_file(CONFIG_PATH, json.dumps(agent_config))
        yield url


@solver
def svgs_solver(
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Solver:
    defaults = load_task_defaults("svgs")
    assert defaults.max_turns is not None
    assert defaults.timeout_s is not None
    solve_episode = sandbox_solver(
        harness, harness_version, harness_model, defaults.max_turns, timeout_s=defaults.timeout_s
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        captured: dict[str, Any] = {}
        try:
            async with leaderboard_endpoint(state.metadata["rubric_path"], captured):
                solved = await solve_episode(state, generate)
        finally:
            if "submissions" in captured:
                state.metadata["submissions"] = captured["submissions"]
            if captured.get("errors"):
                state.metadata["leaderboard_errors"] = captured["errors"]
        if captured.get("errors"):
            raise RuntimeError(f"Leaderboard service failed: {captured['errors'][-1]}")
        return solved

    return solve
