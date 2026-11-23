import json
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate
from tests.cheatbench_inspect.openmath.conftest import FIXTURES

from cheatbench_inspect.openmath.resources.judge_schema import EXTRA_CONTEXT, OpenMathVerdict
from cheatbench_inspect.openmath.scorers import openmath_scorer

VERDICTS = json.loads((FIXTURES / "verdicts.json").read_text())


def test_parity_with_upstream() -> None:
    assert json.loads((FIXTURES / "extra_context.json").read_text()) == EXTRA_CONTEXT
    schema = OpenMathVerdict.model_json_schema()
    expected = json.loads((FIXTURES / "shared_verdict_schema.json").read_text())
    assert schema == expected
    assert list(schema["properties"]) == list(expected["properties"])
    for verdict in VERDICTS.values():
        assert OpenMathVerdict.model_validate(verdict).model_dump() == verdict


@pytest.mark.parametrize(
    "case", ["inconsistent_flags", "missing_field", "invalid_discovery", "shared"]
)
def test_composed_judge_validation(tmp_path: Path, case: str) -> None:
    verdict = VERDICTS["honest"].copy()
    if case == "inconsistent_flags":
        # Like upstream, the schema's only cross-field check is for discovery.
        verdict.update(attempted_extract=True, extract_attempts=1)
    elif case == "missing_field":
        del verdict["listed_archive"]
    elif case == "invalid_discovery":
        verdict.update(honey_pots_seen=False)
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
            scorer=openmath_scorer(),
        ),
        model="mockllm/model",
        model_roles={"grader": grader},
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.samples
    [sample] = log.samples
    if case in {"missing_field", "invalid_discovery"}:
        assert log.status == "error"
        assert sample.error is not None
        error = "listed_archive" if case == "missing_field" else "establishes honey_pots_seen"
        assert error in sample.error.message
    else:
        assert log.status == "success"
        assert sample.scores
        [score] = sample.scores.values()
        assert score.value is (case == "shared")
