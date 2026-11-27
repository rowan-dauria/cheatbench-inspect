"""Stage the research repository, replay the article and export deliverables."""

import io
import tarfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.openmath.replay import NODE_CA, replay_endpoint
from cheatbench_inspect.openmath_agent.build import RESOURCES
from cheatbench_inspect.openmath_agent.dataset import relative_path
from cheatbench_inspect.sandbox_task import sandbox_service_endpoint, sandbox_solver


async def stage_repository(environment_dir: Path, metadata: dict[str, Any]) -> None:
    """Transfer an entire worktree, including Git state, in one sandbox write."""
    files = metadata["repository_files"]
    mtimes = metadata["repository_mtimes"]
    modes = metadata["repository_modes"]
    directories: dict[PurePosixPath, int] = {}
    timestamps = {
        name: int(datetime.fromisoformat(stamp).replace(tzinfo=UTC).timestamp())
        if isinstance(stamp, str)
        else int(stamp)
        for name, stamp in mtimes.items()
    }
    for name, stamp in timestamps.items():
        for parent in relative_path(name).parents:
            if parent.parts:
                directories[parent] = max(stamp, directories.get(parent, stamp))

    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for directory, stamp in sorted(directories.items()):
            info = tarfile.TarInfo(str(directory))
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.mtime = stamp
            archive.addfile(info)

        for dest, source in files.items():
            name = relative_path(dest).as_posix()
            file = environment_dir / relative_path(source)
            info = archive.gettarinfo(str(file), arcname=name)
            if not info.isfile() or file.resolve() != file.absolute():
                raise ValueError(f"Repository file must be a regular file: {file}")
            if dest in timestamps:
                info.mtime = timestamps[dest]
            if (mode := modes.get(dest)) is not None:
                info.mode = int(mode, 8) if isinstance(mode, str) else mode
            with file.open("rb") as stream:
                archive.addfile(info, stream)

    environment = sandbox()
    temporary = await environment.exec(["mktemp", "/var/tmp/.workspace.XXXXXXXX"], user="root")
    if not temporary.success:
        raise RuntimeError(f"Could not allocate repository staging file: {temporary.stderr}")
    path = temporary.stdout.strip()
    try:
        await environment.write_file(path, data.getvalue())
        result = await environment.exec(
            [
                "tar",
                "-xf",
                path,
                "-C",
                "/workspace",
                "--same-permissions",
                "--no-same-owner",
                "--delay-directory-restore",
            ],
            user="root",
        )
        if not result.success:
            raise RuntimeError(f"Could not stage repository: {result.stderr}")
    finally:
        await environment.exec(["rm", "-f", "--", path], user="root")


_CAPTURE = """
import json, sys
# The original runner skips top-level directories created during staging.
print("file:///workspace", flush=True)
sys.stdin.read()
files = collect_deliverables("/workspace", PAYLOAD["skip_dirs"], PAYLOAD["deliverable_files"])
print(json.dumps({"deliverables": files}), flush=True)
"""


@solver
def openmath_agent_solver(
    task_dir: str,
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Solver:
    environment_dir = Path(task_dir) / "environment"
    defaults = load_task_defaults("openmath_agent")
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
        await stage_repository(environment_dir, state.metadata)
        mirror_files = {
            "server.py": (environment_dir / "server.py").read_bytes(),
            "article.html": (environment_dir / "site/article.html").read_bytes(),
        }
        capture = {
            "code": (RESOURCES / "deliverables.py").read_text() + _CAPTURE,
            "skip_dirs": sorted(
                {
                    name.split("/", 1)[0]
                    for name in state.metadata["repository_files"]
                    if "/" in name
                }
            ),
            "deliverable_files": state.metadata["deliverable_files"],
        }
        requests: dict[str, str] = {}
        saved_work: dict[str, Any] = {}
        try:
            async with (
                replay_endpoint(
                    mirror_files,
                    ["openai.com"],
                    "serve_article(Path(PAYLOAD['directory']), PAYLOAD['hosts'])",
                    requests,
                ),
                sandbox_service_endpoint(capture, grader_state=saved_work),
            ):
                return await solve_episode(state, generate)
        finally:
            if "requests" in requests:
                state.store.set("mirror_requests", requests["requests"])
            if "deliverables" in saved_work:
                state.store.set("deliverables", saved_work["deliverables"])

    return solve
