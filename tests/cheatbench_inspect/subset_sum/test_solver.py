from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState

from cheatbench_inspect import sandbox_task
from cheatbench_inspect.harnesses.harness import HarnessResult
from cheatbench_inspect.subset_sum import subset_sum


@pytest.mark.parametrize("harness", ["terminus-2", "claude-code", "codex"])
@pytest.mark.parametrize("fail", [False, True])
async def test_subset_sum_solver(harness: str, fail: bool) -> None:
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="q4",
        epoch=1,
        input="solve",
        messages=[],
        metadata={"numbers": [1, 2], "target": 3, "staged_files": []},
    )
    events = []

    uses_cli = harness != "terminus-2"

    @asynccontextmanager
    async def endpoint(
        instance: tuple[list[int], int], *, task_id: str | None, grader_state: dict | None
    ) -> AsyncGenerator[str, None]:
        assert task_id == ("q4" if uses_cli else None)
        assert grader_state == ({} if uses_cli else None)
        assert instance == ([1, 2], 3)
        events.append("open")
        try:
            yield "http://localhost/mcp"
        finally:
            events.append("close")
            if grader_state is not None:
                grader_state["accepted"] = [0, 1]

    async def run(name, state, policy, options):
        events.append("run")
        assert name == harness
        assert policy.mcp_servers == [
            {"name": "verifier", "transport": "streamable-http", "url": "http://localhost/mcp"}
        ]
        assert options == ({"cli_version": "1", "cli_model": "model"} if uses_cli else None)
        assert policy.max_turns == (None if uses_cli else 150)
        assert policy.timeout_s == 3600
        if fail:
            raise RuntimeError("failure")
        return (
            AgentState(messages=state.messages)
            if uses_cli
            else HarnessResult({"steps": []}, "", None, {})
        )

    with (
        patch.object(subset_sum, "verifier_endpoint", endpoint),
        patch.object(sandbox_task, "run_harness", run),
        patch.object(sandbox_task, "hand_staged_files_to_agent", AsyncMock()),
    ):
        solve = subset_sum.subset_sum_solver(
            harness, "1" if uses_cli else None, "model" if uses_cli else None
        )
        if fail:
            with pytest.raises(RuntimeError, match="failure"):
                await solve(state, AsyncMock())
        else:
            assert await solve(state, AsyncMock()) is state
            assert state.store.get("harness_artifacts") == (
                {"grader_state": {"accepted": [0, 1]}} if uses_cli else {}
            )
    assert events == ["open", "run", "close"]
