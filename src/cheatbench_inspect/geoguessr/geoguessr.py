"""GeoGuessr sandbox solver."""

from inspect_ai.solver import Solver, solver

from cheatbench_inspect.sandbox_task import sandbox_solver

DEFAULT_EPOCHS = 1
MAX_TURNS = 150
# Original tasks/geoguessr/task.yaml:11.
TIMEOUT_S = 3600


@solver
def geoguessr_solver(
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Solver:
    """Solve GeoGuessr with the selected harness and no verifier MCP server."""
    return sandbox_solver(harness, harness_version, harness_model, MAX_TURNS, TIMEOUT_S)
