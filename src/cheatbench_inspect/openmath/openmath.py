"""Start the sample's HTTPS replay before running the selected harness."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from inspect_ai.solver import Generate, Solver, TaskState, solver

from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.openmath.build import RESOURCES, SITE_BUNDLE
from cheatbench_inspect.openmath.dataset import row_responses
from cheatbench_inspect.openmath.replay import NODE_CA, replay_endpoint
from cheatbench_inspect.sandbox_task import sandbox_solver

HOSTS = ["terrytao.wordpress.com", "public-api.wordpress.com"]


@asynccontextmanager
async def mirror_endpoint(
    task_dir: Path, row: str, request_state: dict[str, str] | None = None
) -> AsyncGenerator[str, None]:
    files = row_responses(task_dir, row)
    for module in ("mirror", "inject"):
        files[f"{module}.py"] = (RESOURCES / f"{module}.txt").read_bytes()
    files["capture.tar.xz"] = (task_dir / SITE_BUNDLE).read_bytes()
    async with replay_endpoint(
        files,
        HOSTS,
        "serve(Path(PAYLOAD['directory']), PAYLOAD['row'], PAYLOAD['hosts'])",
        request_state,
        row=row,
    ) as directory:
        yield directory


@solver
def openmath_solver(
    task_dir: str,
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Solver:
    defaults = load_task_defaults("openmath")
    assert defaults.max_turns is not None
    assert defaults.timeout_s is not None
    solve_episode = sandbox_solver(
        harness,
        harness_version,
        harness_model,
        defaults.max_turns,
        timeout_s=defaults.timeout_s,
        agent_env={"NODE_EXTRA_CA_CERTS": NODE_CA},
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        requests: dict[str, str] = {}
        try:
            async with mirror_endpoint(Path(task_dir), str(state.sample_id), requests):
                return await solve_episode(state, generate)
        finally:
            if "requests" in requests:
                state.store.set("mirror_requests", requests["requests"])

    return solve
