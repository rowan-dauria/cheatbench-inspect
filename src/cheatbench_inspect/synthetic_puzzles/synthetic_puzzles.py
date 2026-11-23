"""Synthetic Puzzles sandbox solver."""

from inspect_ai.solver import Solver, solver

from cheatbench_inspect.sandbox_task import sandbox_solver

DEFAULT_EPOCHS = 1
MAX_TURNS = 150
# Original tasks/synthetic_puzzles/task.yaml:3.
TIMEOUT_S = 1800


@solver
def synthetic_puzzles_solver(
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Solver:
    """Solve Synthetic Puzzles with the selected harness and no verifier MCP server."""
    return sandbox_solver(harness, harness_version, harness_model, MAX_TURNS, TIMEOUT_S)
