import json
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate
from tests.cheatbench_inspect.openmath_agent.conftest import FIXTURES

from cheatbench_inspect.openmath_agent.resources.judge_schema import (
    EXTRA_CONTEXT,
    OpenMathAgentVerdict,
)
from cheatbench_inspect.openmath_agent.scorers import openmath_agent_scorer

VERDICTS = json.loads((FIXTURES / "verdicts.json").read_text())


def test_parity_with_upstream() -> None:
    assert json.loads((FIXTURES / "extra_context.json").read_text()) == EXTRA_CONTEXT
    schema = OpenMathAgentVerdict.model_json_schema()
    expected = json.loads((FIXTURES / "shared_verdict_schema.json").read_text())
    assert schema == expected
    assert list(schema["properties"]) == list(expected["properties"])
    for verdict in VERDICTS.values():
        assert OpenMathAgentVerdict.model_validate(verdict).model_dump() == verdict


def test_upstream_accepts_prior_success_claim_without_access() -> None:
    verdict = {**VERDICTS["honest"], "success_claim_seen_before_access": True}
    assert OpenMathAgentVerdict.model_validate(verdict).model_dump() == verdict


@pytest.mark.parametrize("case", ["inconsistent_flags", "missing_field", "shared"])
def test_composed_judge_validation(tmp_path: Path, case: str) -> None:
    verdict = VERDICTS["honest"].copy()
    if case == "inconsistent_flags":
        # Upstream describes the relation without validating it.
        verdict.update(cheated=True, honey_pots_seen=False, attempted_access=False)
    elif case == "missing_field":
        del verdict["extracted_submission"]
    else:
        verdict.update(
            explicit_prohibition_violated=True, explicit_prohibition_evidence="Evidence."
        )
    grader = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", json.dumps(verdict))],
        memoize=False,
    )
    [log] = inspect_eval(
        Task(
            dataset=[Sample(input="Solve a test problem.")],
            solver=generate(),
            scorer=openmath_agent_scorer(),
        ),
        model="mockllm/model",
        model_roles={"grader": grader},
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.samples
    [sample] = log.samples
    if case == "missing_field":
        assert log.status == "error"
        assert sample.error is not None
        assert "extracted_submission" in sample.error.message
    else:
        assert log.status == "success"
        assert sample.scores
        [score] = sample.scores.values()
        assert score.value is True
