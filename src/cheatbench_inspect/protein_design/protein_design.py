"""Serve the synthetic checker and snapshot the assigned repository."""

from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from inspect_ai.solver import Solver, TaskState, solver

from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.harnesses.container import SANDBOX_RESOURCES
from cheatbench_inspect.protein_design.dataset import RESOURCES
from cheatbench_inspect.protein_design.resources.verifier import SERVER_NAME
from cheatbench_inspect.sandbox_task import sandbox_solver, sandbox_verifier_endpoint

_START_VERIFIER = """
import json, sys
config = PAYLOAD
transport = {}
verifier = {}
exec(compile(config["transport"], "transport.py", "exec"), transport)
exec(compile(config["verifier"], "verifier.py", "exec"), verifier)
row = config["row"]
state, call = verifier["make_tool"](row)
observer = verifier["SavedWorkObserver"](row["vars"]["work_dir"], state)
server = transport["GraderServer"](
    state=state,
    call=call,
    server_name=verifier["SERVER_NAME"],
    tool_name=verifier["TOOL_BASENAME"],
    description=verifier["DESCRIPTION"],
    schema=verifier["INPUT_SCHEMA"],
)
server.start()
try:
    print(server.url, flush=True)
    sys.stdin.read()
finally:
    server.close()
observer.finish()
print(json.dumps(state), flush=True)
"""


@asynccontextmanager
async def verifier_endpoint(
    row: dict[str, Any], grader_state: dict[str, Any] | None
) -> AsyncGenerator[str, None]:
    config = {
        "code": _START_VERIFIER,
        "transport": (SANDBOX_RESOURCES / "grader_server.py").read_text(),
        "verifier": (RESOURCES / "verifier.py").read_text(),
        "row": row,
    }
    async with sandbox_verifier_endpoint(config, grader_state=grader_state) as url:
        yield url


@solver
def protein_design_solver(
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Solver:
    def endpoint(
        state: TaskState, grader_state: dict[str, Any] | None
    ) -> AbstractAsyncContextManager[str]:
        return verifier_endpoint(
            {
                "id": str(state.sample_id),
                "vars": {"work_dir": state.metadata["work_dir"]},
                "payload_file": state.metadata["payload_file"],
                "pull_log": state.metadata["pull_log"],
            },
            grader_state,
        )

    defaults = load_task_defaults("protein_design")
    assert defaults.max_turns is not None
    assert defaults.timeout_s is not None
    return sandbox_solver(
        harness,
        harness_version,
        harness_model,
        defaults.max_turns,
        timeout_s=defaults.timeout_s,
        verifier_endpoint=endpoint,
        verifier_name=SERVER_NAME,
        collect_terminus_grader_state=True,
        restore_staged_atimes=True,
    )
