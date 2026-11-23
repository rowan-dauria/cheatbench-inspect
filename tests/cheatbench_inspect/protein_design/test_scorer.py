import json
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate
from tests.cheatbench_inspect.protein_design.conftest import read_fixture

from cheatbench_inspect.protein_design.scorers import protein_design_scorer


@pytest.mark.parametrize("case", ["inconsistent", "shared_prohibition", "missing_field"])
def test_judge_validates_composed_verdict(tmp_path: Path, case: str) -> None:
    verdict = read_fixture("verdicts.json")["honest"]
    if case == "missing_field":
        verdict.pop("checks_used")
    elif case == "shared_prohibition":
        verdict["explicit_prohibition_violated"] = True
    else:
        verdict["read_sequence_file"] = True

    task = Task(
        dataset=[Sample(input="Run the scripted promotion work test.")],
        solver=generate(),
        scorer=protein_design_scorer(),
    )
    grader = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", json.dumps(verdict))],
        memoize=False,
    )
    [log] = inspect_eval(
        task,
        model="mockllm/model",
        model_roles={"grader": grader},
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.samples
    [sample] = log.samples
    if case != "missing_field":
        assert log.status == "success"
        assert sample.scores
        [score] = sample.scores.values()
        assert score.value is (case == "shared_prohibition")
        assert score.explanation == verdict["summary"]
    else:
        assert log.status == "error"
        assert sample.error is not None
        assert "SharedVerdict" in sample.error.message
        assert "checks_used" in sample.error.message
