"""Prepare SWE-bench Pro sandboxes, run the selected harness and verify correctness."""

from __future__ import annotations

import shlex
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from anyio import CancelScope
from inspect_ai.agent import AgentState
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.harnesses.container import ROOT_USER
from cheatbench_inspect.harnesses.harness import HarnessResult
from cheatbench_inspect.sandbox_task import sandbox_solver
from cheatbench_inspect.software_engineering.verifier import (
    VERIFIER_TIMEOUT_S,
    run_correctness_verifier,
)

# Same pinned Harbor commit and tarball SHA-256 as harnesses/resources/sandbox/Dockerfile.
_HARBOR_COMMIT = "4641fd590f190411c57e1cbd824a9f6786cbe92e"
_HARBOR_SHA256 = "ca4079f8189f290134c8619c66a131b21a598893e4d454cceb64a8f53b8cfc79"
_UV_VERSION = "0.9.9"

_INSTALL_LINUX_HEADERS = """
set -eu
if command -v apk >/dev/null 2>&1; then
  apk add --no-cache linux-headers
else
  apt-get -o Acquire::Check-Valid-Until=false update -qq || true
  apt-get install -y --no-install-recommends linux-libc-dev
  rm -rf /var/lib/apt/lists/*
fi
"""

_BOOTSTRAP_TERMINUS_RUNTIME = f"""
set -eu
if [ -x /opt/terminus-runtime/.venv/bin/python ]; then
  exit 0
fi
if ! command -v tmux >/dev/null 2>&1; then
  if command -v apk >/dev/null 2>&1; then
    apk add --no-cache tmux curl ca-certificates
  else
    apt-get -o Acquire::Check-Valid-Until=false update -qq || true
    apt-get install -y --no-install-recommends tmux curl ca-certificates
    rm -rf /var/lib/apt/lists/*
  fi
fi
mkdir -p /opt/terminus-runtime /tmp/harbor-src
if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL "https://astral.sh/uv/{_UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/opt/terminus-runtime/bin sh
  export PATH="/opt/terminus-runtime/bin:$PATH"
fi
cd /tmp/harbor-src
curl -fsSL --retry 3 "https://codeload.github.com/harbor-framework/harbor/tar.gz/{_HARBOR_COMMIT}" -o source.tar.gz
echo "{_HARBOR_SHA256}  source.tar.gz" | sha256sum -c -
tar -xzf source.tar.gz --strip-components=1
UV_PYTHON_INSTALL_DIR=/opt/terminus-runtime/python \\
  UV_PROJECT_ENVIRONMENT=/opt/terminus-runtime/.venv \\
  uv sync --frozen --no-dev --no-editable --package harbor --python 3.12.11
uv pip install --python /opt/terminus-runtime/.venv/bin/python --no-deps pillow==11.3.0 asciinema==2.4.0
ln -sf /opt/terminus-runtime/.venv/bin/asciinema /usr/local/bin/asciinema
cd /
rm -rf /tmp/harbor-src /root/.cache/uv
chmod -R a+rX /opt/terminus-runtime
"""


async def _noop_stage(_staged: list[str]) -> None:
    return None


async def prepare_instance_sandbox(state: TaskState, *, harness: str) -> None:
    """Reproduce the Dockerfile's post-base setup inside the digest-pinned base image."""
    box = sandbox()
    base_commit = state.metadata["base_commit"]
    reset_cmd = (
        "set -eu; mkdir -p /logs; cd /app; "
        "git config --global --add safe.directory /app; "
        f"git reset --hard {shlex.quote(base_commit)}; "
        "git clean -fd; "
        f"git checkout {shlex.quote(base_commit)}"
    )
    reset = await box.exec(["sh", "-c", reset_cmd], user=ROOT_USER, timeout=300)
    if not reset.success:
        raise RuntimeError(f"Could not reset /app to {base_commit}: {reset.stderr}")

    if state.metadata.get("install_linux_headers"):
        headers = await box.exec(["sh", "-c", _INSTALL_LINUX_HEADERS], user=ROOT_USER, timeout=300)
        if not headers.success:
            raise RuntimeError(f"Could not install linux headers: {headers.stderr}")

    if harness == "terminus-2":
        bootstrap = await box.exec(
            ["sh", "-c", _BOOTSTRAP_TERMINUS_RUNTIME], user=ROOT_USER, timeout=600
        )
        if not bootstrap.success:
            raise RuntimeError(f"Could not bootstrap Terminus-2 runtime: {bootstrap.stderr}")


@solver
def software_engineering_solver(
    task_dir: str,
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
    *,
    verifier_timeout_s: int = VERIFIER_TIMEOUT_S,
    prepare_sandbox: Callable[..., Awaitable[None]] | None = None,
    run_verifier: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    launch_harness: Callable[..., Awaitable[HarnessResult | AgentState]] | None = None,
) -> Solver:
    """Set up the SWE-bench Pro repo, run the harness, then run tests/test.sh."""
    defaults = load_task_defaults("software_engineering")
    assert defaults.max_turns is not None
    assert defaults.timeout_s is not None
    solve_episode = sandbox_solver(
        harness,
        harness_version,
        harness_model,
        defaults.max_turns,
        timeout_s=defaults.timeout_s,
        agent_env={"PYTHONPATH": "/app/lib:/app"},
        cwd="/app",
        allow_web_tools=True,
        user=ROOT_USER,
        stage_files=_noop_stage,
        launch_harness=launch_harness,
    )
    prepare = prepare_sandbox or prepare_instance_sandbox
    verify = run_verifier or run_correctness_verifier

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await prepare(state, harness=harness)
        launch_error: BaseException | None = None
        try:
            return await solve_episode(state, generate)
        except BaseException as error:
            launch_error = error
            raise
        finally:
            try:
                with CancelScope(shield=True):
                    report = await verify(
                        Path(task_dir), str(state.sample_id), timeout_s=verifier_timeout_s
                    )
                    state.metadata["correctness"] = report
                    state.store.set("correctness", report)
            except Exception as error:
                if launch_error is None:
                    fallback = {
                        "resolved": False,
                        "reward": 0,
                        "exit_code": None,
                        "suite_load_error": True,
                        "assertion_failure": False,
                        "verifier_error": f"{type(error).__name__}: {error}",
                    }
                    state.metadata["correctness"] = fallback
                    state.store.set("correctness", fallback)
                else:
                    launch_error.add_note(f"Post-run correctness verification failed: {error}")

    return solve
