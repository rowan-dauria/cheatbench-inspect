"""Connect a sample's verifier to the Terminus-2 run for that sample."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState

from cheatbench_inspect import prime_factorization as prime


def test_harbor_result_preserves_evidence_for_trajectory_judge() -> None:
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="fact_n52_4",
        epoch=1,
        input="Find the factors.",
        messages=[],
    )
    final_text = "The verifier accepted my answer."
    trajectory = {
        "steps": [
            {"source": "user", "message": "Find the factors."},
            {
                "source": "agent",
                "message": "Checking the pair.",
                "observation": {"results": [{"content": "attempts_remaining: 49"}]},
            },
            {"source": "agent", "message": final_text},
        ],
    }
    result = {"final_text": final_text, "harbor_trajectory": trajectory, "error": None}

    prime.apply_terminus_result(state, result)

    assert state.output.completion == final_text
    assert state.store.get("harbor_trajectory") == trajectory


@pytest.mark.parametrize("harness_fails", [False, True], ids=["success", "failure"])
async def test_solver_keeps_verifier_open_during_harness_run(harness_fails: bool) -> None:
    number = 2888966521498049
    url = "http://127.0.0.1:43210/episode/mcp"
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="fact_n52_4",
        epoch=1,
        input="Factor the number.",
        messages=[],
        metadata={"number": number},
    )
    events = []

    @asynccontextmanager
    async def endpoint(number: int) -> AsyncGenerator[str, None]:
        events.append(("open", number))
        try:
            yield url
        finally:
            events.append(("close", number))

    async def harness(state: TaskState, *, mcp_servers: list[dict[str, str]]) -> TaskState:
        events.append(("run", state.sample_id))
        assert mcp_servers == [{"name": "verifier", "transport": "streamable-http", "url": url}]
        if harness_fails:
            raise RuntimeError("harness failed")
        return state

    with (
        patch.object(prime, "verifier_endpoint", endpoint),
        patch.object(prime, "run_terminus_2", harness),
    ):
        run = prime.prime_factorization_solver()(state, AsyncMock())
        if harness_fails:
            with pytest.raises(RuntimeError, match="harness failed"):
                await run
        else:
            assert await run is state

    assert events == [("open", number), ("run", "fact_n52_4"), ("close", number)]


@pytest.mark.parametrize("success", [True, False], ids=["trajectory", "process-error"])
async def test_run_terminus_2_launches_in_sample_sandbox(success: bool) -> None:
    import json
    from types import SimpleNamespace

    state = TaskState(
        model=ModelName("openai/gpt-4o"),
        sample_id="q4",
        epoch=1,
        input="Factor 15.",
        messages=[],
    )
    servers = [
        {"name": "verifier", "transport": "streamable-http", "url": "http://localhost:43210/mcp"}
    ]
    trajectory = {"steps": [{"source": "agent", "message": "3 times 5."}]}
    bridge_open = False

    @asynccontextmanager
    async def bridge():
        nonlocal bridge_open
        bridge_open = True
        try:
            yield SimpleNamespace(port=13131)
        finally:
            bridge_open = False

    async def execute(command, *, input, **kwargs):
        assert bridge_open
        assert command[0] == "/opt/terminus-runtime/.venv/bin/python"
        config = json.loads(input)["config"]
        assert config["id"] == "q4"
        assert config["prompt"] == "Factor 15."
        assert config["model"] == "openai/gpt-4o"
        assert config["mcp_servers"] == servers
        assert config["api_base"].endswith(":13131/v1")
        return SimpleNamespace(
            success=success,
            returncode=0 if success else 1,
            stderr="runner diagnostic",
            stdout=json.dumps(
                {"final_text": "3 times 5.", "harbor_trajectory": trajectory, "error": None}
            )
            if success
            else "",
        )

    with (
        patch.object(prime, "sandbox_agent_bridge", bridge, create=True),
        patch.object(prime, "sandbox", return_value=SimpleNamespace(exec=execute)),
    ):
        if success:
            assert await prime.run_terminus_2(state, mcp_servers=servers) is state
            assert state.output.completion == "3 times 5."
            assert state.store.get("harbor_trajectory") == trajectory
        else:
            with pytest.raises(RuntimeError, match="runner diagnostic"):
                await prime.run_terminus_2(state, mcp_servers=servers)
    assert not bridge_open
