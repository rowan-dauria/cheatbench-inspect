"""Start the Knowledge Work review tool before launching the agent."""

from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from inspect_ai.solver import Solver, TaskState, solver

from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.harnesses.container import SANDBOX_RESOURCES
from cheatbench_inspect.knowledge_work.dataset import RESOURCES
from cheatbench_inspect.sandbox_task import sandbox_solver, sandbox_verifier_endpoint

# LibreOffice alone has a 90 s timeout. Allow time to rasterize its PDF as well.
REVIEW_STARTUP_S = 120
SERVER_NAME = "review"

_START_VERIFIER = """
import json, sys
# PyMuPDF may emit diagnostics to stdout while rendering. Only send the
# endpoint and state snapshot over this stream.
protocol_stdout = sys.stdout
sys.stdout = sys.stderr
config = PAYLOAD
transport = {}
verifier = {}
exec(compile(config["transport"], "transport.py", "exec"), transport)
exec(compile(config["verifier"], "verifier.py", "exec"), verifier)
destination = config["reference_destination"]
row = {"id": config["task_id"], "reference_destination": destination}
state, call = verifier["make_tool"](row, {destination: destination}, "/workspace")
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
    print(server.url, file=protocol_stdout, flush=True)
    sys.stdin.read()
finally:
    server.close()
print(json.dumps(state), file=protocol_stdout, flush=True)
"""


@asynccontextmanager
async def verifier_endpoint(
    task_id: str, reference_destination: str, grader_state: dict[str, Any] | None
) -> AsyncGenerator[str, None]:
    config = {
        "code": _START_VERIFIER,
        "transport": (SANDBOX_RESOURCES / "grader_server.py").read_text(),
        "verifier": (RESOURCES / "verifier.py").read_text(),
        "task_id": task_id,
        "reference_destination": reference_destination,
    }
    async with sandbox_verifier_endpoint(
        config, grader_state=grader_state, startup_timeout_s=REVIEW_STARTUP_S
    ) as url:
        yield url


@solver
def knowledge_work_solver(
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Solver:
    def endpoint(
        state: TaskState, grader_state: dict[str, Any] | None
    ) -> AbstractAsyncContextManager[str]:
        return verifier_endpoint(
            str(state.sample_id), state.metadata["reference_destination"], grader_state
        )

    defaults = load_task_defaults("knowledge_work")
    assert defaults.max_turns is not None
    assert defaults.timeout_s is not None
    return sandbox_solver(
        harness,
        harness_version,
        harness_model,
        defaults.max_turns,
        defaults.timeout_s,
        endpoint,
        verifier_name=SERVER_NAME,
        collect_terminus_grader_state=True,
    )
