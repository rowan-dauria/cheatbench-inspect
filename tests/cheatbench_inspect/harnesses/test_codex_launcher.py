"""Host-side Codex route, configuration and harness dispatch tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.model import GenerateConfig, Model, ModelName, get_model
from inspect_ai.solver import TaskState

import cheatbench_inspect.harnesses.codex as codex
from cheatbench_inspect.harnesses.harness import (
    HarnessPolicy,
    run_harness,
    validate_harness,
)
from cheatbench_inspect.tasks import cheatbench_prime_factorization


def _state(
    *,
    sample_id: str = "pf-15",
    input_text: str = "Factor 15.",
    metadata: dict[str, Any] | None = None,
) -> TaskState:
    return TaskState(
        model=ModelName("openrouter/openai/gpt-5.6-luna"),
        sample_id=sample_id,
        epoch=1,
        input=input_text,
        messages=[],
        metadata=metadata if metadata is not None else {"number": 15, "staged_files": []},
    )


def test_validate_responses_route_accepts_openrouter_and_openai_responses() -> None:
    or_responses = get_model(
        "openrouter/openai/gpt-5.6-luna",
        base_url="http://127.0.0.1:1/api/v1",
        api_key="dummy",
        responses_api=True,
    )
    codex.validate_responses_route(or_responses)

    openai_gpt5 = get_model("openai/gpt-5", api_key="dummy")
    codex.validate_responses_route(openai_gpt5)

    openai_gpt4o_explicit = get_model("openai/gpt-4o", api_key="dummy", responses_api=True)
    codex.validate_responses_route(openai_gpt4o_explicit)

    # Fallback provider with only public `responses_api` attribute (no `uses_responses_api`).
    attr_only = cast(
        Model,
        SimpleNamespace(api=SimpleNamespace(responses_api=True), config=GenerateConfig()),
    )
    codex.validate_responses_route(attr_only)


@pytest.mark.parametrize(
    "make_model",
    [
        lambda: get_model(
            "openrouter/openai/gpt-5.6-luna",
            base_url="http://127.0.0.1:1/api/v1",
            api_key="dummy",
        ),
        lambda: get_model(
            "openrouter/openai/gpt-5.6-luna",
            base_url="http://127.0.0.1:1/api/v1",
            api_key="dummy",
            responses_api=False,
        ),
        lambda: get_model("openai/gpt-4o", api_key="dummy"),
        lambda: get_model("openai/gpt-5", api_key="dummy", responses_api=False),
        lambda: get_model("mockllm/model"),
        lambda: cast(
            Model,
            SimpleNamespace(api=SimpleNamespace(responses_api=False), config=GenerateConfig()),
        ),
    ],
    ids=[
        "openrouter-default",
        "openrouter-false",
        "openai-gpt4o-default",
        "openai-gpt5-false",
        "mockllm",
        "attr-only-false",
    ],
)
def test_validate_responses_route_rejects_non_responses_models(make_model: Any) -> None:
    with pytest.raises(ValueError, match="Responses API"):
        codex.validate_responses_route(make_model())


# ---------------------------------------------------------------------------
# Payload and run_codex lifecycle
# ---------------------------------------------------------------------------


async def test_harness_codex_dispatch_and_option_validation() -> None:
    validate_harness("codex")
    state = _state()
    policy = HarnessPolicy([], 10, 60)
    expected = AgentState(messages=[])

    with patch.object(codex, "run_codex", AsyncMock(return_value=expected)) as run:
        result = await run_harness(
            "codex", state, policy, {"cli_version": "0.154.0", "cli_model": "gpt-5.6-luna"}
        )
        assert result is expected
        run.assert_awaited_once()
        assert isinstance(run.call_args.args[0], AgentState)
        assert run.call_args.args[1:] == (policy,)
        assert run.call_args.kwargs == {"cli_version": "0.154.0", "cli_model": "gpt-5.6-luna"}

    with pytest.raises(ValueError, match="codex requires cli_version and cli_model"):
        await run_harness("codex", state, policy)
    with pytest.raises(ValueError, match="codex requires cli_version and cli_model"):
        await run_harness("codex", state, policy, {"cli_version": "0.154.0"})
    with pytest.raises(ValueError, match="codex requires cli_version and cli_model"):
        await run_harness("codex", state, policy, {"cli_model": "gpt-5.6-luna"})
    with pytest.raises(ValueError, match="Unknown options for codex: extra"):
        await run_harness(
            "codex",
            state,
            policy,
            {"cli_version": "0.154.0", "cli_model": "gpt-5.6-luna", "extra": "1"},
        )
    with pytest.raises(ValueError, match="Unknown options for claude-code: extra"):
        await run_harness(
            "claude-code",
            state,
            policy,
            {"cli_version": "2.1.270", "cli_model": "claude-sonnet-5", "extra": "1"},
        )


def test_prime_task_codex_argument_validation_and_compose(tmp_path: Path) -> None:
    (tmp_path / "environment").mkdir()
    (tmp_path / "data.jsonl").write_text(
        json.dumps(
            {
                "id": "pf-15",
                "number": 15,
                "input": "Factor 15.",
                "files": {},
                "files_abs": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="codex requires harness_version and harness_model"):
        cheatbench_prime_factorization(task_dir=str(tmp_path), harness="codex")
    with pytest.raises(ValueError, match="codex requires harness_version and harness_model"):
        cheatbench_prime_factorization(
            task_dir=str(tmp_path), harness="codex", harness_version="0.154.0"
        )
    with pytest.raises(ValueError, match="codex requires harness_version and harness_model"):
        cheatbench_prime_factorization(
            task_dir=str(tmp_path), harness="codex", harness_model="gpt-5.6-luna"
        )

    task = cheatbench_prime_factorization(
        task_dir=str(tmp_path),
        dataset_limit=1,
        harness="codex",
        harness_version="0.154.0",
        harness_model="gpt-5.6-luna",
    )
    assert task.sandbox is not None
    assert task.sandbox.type == "docker"
    assert isinstance(task.sandbox.config, str)
    assert task.sandbox.config.endswith("resources/sandbox/codex-compose.yaml")


@pytest.mark.parametrize("summary", ["detailed", "none", None])
def test_codex_cli_reasoning_overrides(summary: Literal["detailed", "none"] | None) -> None:
    policy = HarnessPolicy([], 10, 60)
    overrides = codex._config_overrides(
        policy, GenerateConfig(reasoning_effort="medium", reasoning_summary=summary)
    )
    assert overrides["model_reasoning_effort"] == '"medium"'
    assert overrides.get("model_reasoning_summary") == (
        '"detailed"' if summary == "detailed" else None
    )
