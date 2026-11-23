"""GeoGuessr sandbox solver."""

from inspect_ai.solver import Solver, solver

from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.sandbox_task import sandbox_solver


@solver
def geoguessr_solver(
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Solver:
    """Solve GeoGuessr with the selected harness and no verifier MCP server."""
    defaults = load_task_defaults("geoguessr")
    assert defaults.max_turns is not None
    assert defaults.timeout_s is not None
    return sandbox_solver(
        harness,
        harness_version,
        harness_model,
        defaults.max_turns,
        defaults.timeout_s,
    )
