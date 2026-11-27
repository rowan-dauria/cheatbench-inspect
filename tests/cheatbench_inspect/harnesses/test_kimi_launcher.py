"""Kimi Code launcher and dispatch."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState
from inspect_ai.tool import MCPServerConfigHTTP

import cheatbench_inspect.harnesses.kimi_code as launcher
from cheatbench_inspect.harnesses.harness import HarnessPolicy, run_harness, validate_harness


@pytest.mark.parametrize("max_turns", [None, 150])
async def test_launcher(max_turns: int | None) -> None:
    state = AgentState(messages=[])
    policy = HarnessPolicy(
        [{"name": "verifier", "url": "http://localhost:123/mcp"}],
        max_turns=max_turns,
        env={"NODE_EXTRA_CA_CERTS": "/tmp/router.pem"},
    )
    agent = AsyncMock(return_value=state)
    box = MagicMock()
    box.exec = AsyncMock(return_value=MagicMock(success=True))
    box.write_file = AsyncMock()
    previous_port = 3114 if max_turns else None
    sample_store = {} if previous_port is None else {"kimi_code_model_port": previous_port}
    with (
        patch.object(launcher, "kimi_code", return_value=agent) as factory,
        patch.object(launcher, "sandbox", return_value=box),
        patch.object(launcher, "store", return_value=sample_store),
    ):
        assert (
            await launcher.run_kimi_code(
                state, policy, cli_version="0.43.0", cli_model="moonshotai/kimi-k3"
            )
            is state
        )
    env = {
        "HOME": "/home/user",
        "USER": "user",
        "LOGNAME": "user",
        "NODE_EXTRA_CA_CERTS": "/tmp/router.pem",
        "KIMI_CODE_HOME": "/opt/kimi",
        "KIMI_MODEL_NAME": "moonshotai/kimi-k3",
        "KIMI_MODEL_PROVIDER_TYPE": "kimi",
        "KIMI_MODEL_API_KEY": "sk-none",
        "KIMI_MODEL_BASE_URL": f"http://localhost:{(previous_port or 3100) + 1}/v1",
        "KIMI_DISABLE_TELEMETRY": "1",
        "KIMI_CODE_NO_AUTO_UPDATE": "1",
        "NO_COLOR": "1",
    }
    if max_turns is not None:
        env["KIMI_LOOP_MAX_STEPS_PER_TURN"] = str(max_turns)
    factory.assert_called_once_with(
        version="0.43.0",
        model="moonshotai/kimi-k3",
        max_context_size=262_144,
        cwd="/workspace",
        user="user",
        env=env,
        attempts=1,
        retry_refusals=0,
        mcp_servers=[
            MCPServerConfigHTTP(name="verifier", url="http://localhost:123/mcp", type="http")
        ],
    )
    agent.assert_awaited_once_with(state)
    box.write_file.assert_awaited_once_with(
        "/opt/kimi/config.toml",
        'telemetry = false\n\n[tools]\ndisabled = ["FetchURL", "WebSearch"]\n',
    )


async def test_mcp_only_rejected() -> None:
    with pytest.raises(ValueError, match="cannot restrict"):
        await launcher.run_kimi_code(
            AgentState(messages=[]),
            HarnessPolicy([], mcp_only=True),
            cli_version="0.43.0",
            cli_model="moonshotai/kimi-k3",
        )


async def test_dispatch_and_validation() -> None:
    validate_harness("kimi-code")
    state = TaskState(
        model=ModelName("mockllm/model"), sample_id="1", epoch=1, input="test", messages=[]
    )
    policy = HarnessPolicy([])
    result = AgentState(messages=[])
    with patch.object(launcher, "run_kimi_code", AsyncMock(return_value=result)) as launch:
        assert (
            await run_harness(
                "kimi-code",
                state,
                policy,
                {"cli_version": "0.43.0", "cli_model": "moonshotai/kimi-k3"},
            )
            is result
        )
        launch.assert_awaited_once()
    for options in (None, {"cli_version": "0.43.0"}, {"cli_model": "moonshotai/kimi-k3"}):
        with pytest.raises(ValueError, match="requires cli_version and cli_model"):
            await run_harness("kimi-code", state, policy, options)
    with pytest.raises(ValueError, match="Unknown options"):
        await run_harness(
            "kimi-code",
            state,
            policy,
            {"cli_version": "0.43.0", "cli_model": "moonshotai/kimi-k3", "extra": "x"},
        )


@pytest.mark.parametrize(
    "options", [{}, {"harness_version": "0.43.0"}, {"harness_model": "moonshotai/kimi-k3"}]
)
def test_task_requires_kimi_options(options: dict[str, Any]) -> None:
    from cheatbench_inspect.tasks import cheatbench_prime_factorization

    with pytest.raises(ValueError, match="kimi-code requires harness_version and harness_model"):
        cheatbench_prime_factorization(harness="kimi-code", **options)
