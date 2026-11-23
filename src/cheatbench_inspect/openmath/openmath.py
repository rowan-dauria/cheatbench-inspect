"""Start the sample's HTTPS replay before running the selected harness."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.openmath.build import RESOURCES, SITE_BUNDLE
from cheatbench_inspect.openmath.dataset import row_responses
from cheatbench_inspect.sandbox_task import sandbox_service_endpoint, sandbox_solver

HOSTS = ["terrytao.wordpress.com", "public-api.wordpress.com"]
NODE_CA = "/usr/local/share/ca-certificates/site-root.crt"
STARTUP_S = 120


@asynccontextmanager
async def mirror_endpoint(
    task_dir: Path, row: str, request_state: dict[str, str] | None = None
) -> AsyncGenerator[str, None]:
    environment = sandbox()
    result = await environment.exec(["mktemp", "-d", "/var/tmp/.site.XXXXXXXX"], user="root")
    if not result.success:
        raise RuntimeError(f"Could not create the site directory: {result.stderr}")
    directory = result.stdout.strip()
    files = row_responses(task_dir, row)
    for module in ("mirror", "inject"):
        files[f"{module}.py"] = (RESOURCES / f"{module}.txt").read_bytes()
    for name, content in files.items():
        await environment.write_file(f"{directory}/{name}", content)
    await environment.write_file(
        f"{directory}/capture.tar.xz", (task_dir / SITE_BUNDLE).read_bytes()
    )
    config = {
        "code": (RESOURCES / "start_site.py").read_text()
        + "\nserve(Path(PAYLOAD['directory']), PAYLOAD['row'], PAYLOAD['hosts'])",
        "directory": directory,
        "row": row,
        "hosts": HOSTS,
    }
    async with sandbox_service_endpoint(
        config, startup_timeout_s=STARTUP_S, user="root", grader_state=request_state
    ):
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
