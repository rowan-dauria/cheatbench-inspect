"""Stage the original checker and connect it to an episode-scoped reviewer."""

import json
from secrets import token_hex

import anyio
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox, sandbox_service
from pydantic import JsonValue

from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.creative_writings.dataset import RESOURCES
from cheatbench_inspect.creative_writings.reviewer import (
    DEFAULT_REVIEWER_EFFORT,
    DEFAULT_REVIEWER_MODEL,
    Reviewer,
)
from cheatbench_inspect.sandbox_task import sandbox_service_endpoint, sandbox_solver

REVIEW_FILE = "/workspace/grading/.review"


@solver
def creative_writings_solver(
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Solver:
    defaults = load_task_defaults("creative_writings")
    assert defaults.max_turns is not None
    assert defaults.timeout_s is not None
    solve_episode = sandbox_solver(
        harness, harness_version, harness_model, defaults.max_turns, defaults.timeout_s
    )
    proxy = (RESOURCES / "review_proxy.py").read_text(encoding="utf-8")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        environment = sandbox()
        result = await environment.exec(["uname", "-m"])
        extensions = state.metadata["score_extensions"]
        arch = result.stdout.strip()
        if not result.success or arch not in extensions:
            raise RuntimeError(f"No compiled similarity helper for sandbox architecture {arch!r}")
        extension = anyio.Path(extensions[arch])
        await environment.write_file(
            f"/workspace/grading/{extension.name}", await extension.read_bytes()
        )

        calls = state.metadata["review_calls"] = []
        reference = await environment.read_file(state.metadata["reference_path"])
        model = get_model(
            role="reviewer",
            default=DEFAULT_REVIEWER_MODEL,
            config=GenerateConfig(reasoning_effort=DEFAULT_REVIEWER_EFFORT),
        )
        reviewer = Reviewer(model, reference, calls)
        state.metadata["review_errors"] = reviewer.errors

        started = anyio.Event()
        done = anyio.Event()

        async def route_reviews(cancel_scope: anyio.CancelScope) -> None:
            async def review(brief: str, reference: str, submission: str) -> JsonValue:
                reply = await reviewer.review(brief, reference, submission)
                # The reviewer has recorded the failed call before we stop the harness.
                if reviewer.errors:
                    cancel_scope.cancel()
                return reply

            await sandbox_service(
                "editorial",
                [review],
                done.is_set,
                environment,
                user="root",
                started=started,
            )

        solved = state
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(route_reviews, group.cancel_scope)
                try:
                    with anyio.fail_after(30):
                        await started.wait()
                    token = token_hex(16)
                    config = {"code": proxy + '\nserve(PAYLOAD["token"])', "token": token}
                    async with sandbox_service_endpoint(config, user="root") as url:
                        await environment.write_file(
                            REVIEW_FILE, json.dumps({"url": url, "token": token})
                        )
                        solved = await solve_episode(state, generate)
                finally:
                    done.set()
        except Exception as exc:
            if reviewer.errors:
                raise RuntimeError(f"Reviewer failed: {reviewer.errors[0]}") from exc
            raise
        if reviewer.errors:
            raise RuntimeError(f"Reviewer failed: {reviewer.errors[0]}")
        return solved

    return solve
