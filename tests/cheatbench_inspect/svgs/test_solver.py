from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from inspect_ai.solver import TaskState
from tests.cheatbench_inspect.svgs.conftest import RUBRIC

from cheatbench_inspect.svgs import svgs


@pytest.mark.asyncio
async def test_svgs_solver_preserves_submissions_on_agent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = {"number": 1, "rank": 6, "sha256": "a" * 64}

    @asynccontextmanager
    async def endpoint(rubric: str, captured: dict):
        assert rubric == RUBRIC
        try:
            yield "http://127.0.0.1:1234/submit"
        finally:
            captured.update(submissions=[submission], errors=[])

    agent = AsyncMock(side_effect=RuntimeError("agent failed"))
    monkeypatch.setattr(svgs, "leaderboard_endpoint", endpoint)
    monkeypatch.setattr(svgs, "sandbox_solver", Mock(return_value=agent))
    state = Mock(spec=TaskState, metadata={"rubric_path": RUBRIC})
    generate = AsyncMock()
    solve = svgs.svgs_solver()
    with pytest.raises(RuntimeError, match="agent failed"):
        await solve(state, generate)
    assert state.metadata["submissions"] == [submission]
    assert "leaderboard_errors" not in state.metadata
    agent.assert_awaited_once_with(state, generate)
