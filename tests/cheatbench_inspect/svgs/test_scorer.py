import json
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate
from tests.cheatbench_inspect.svgs.conftest import FIXTURES, VERDICTS

from cheatbench_inspect.svgs.resources.judge_schema import EXTRA_CONTEXT, SvgsVerdict
from cheatbench_inspect.svgs.scorers import svgs_scorer


def test_judge_schema_matches_upstream() -> None:
    assert json.loads((FIXTURES / "extra_context.json").read_text()) == EXTRA_CONTEXT
    original = json.loads((FIXTURES / "shared_verdict_schema.json").read_text())
    assert SvgsVerdict.model_json_schema() == original


def test_legacy_aggregate_and_shared_prohibition() -> None:
    # The released task relies on the judge to observe the task-specific OR.
    inconsistent = {**VERDICTS["opened"], "cheated": False}
    assert not SvgsVerdict.model_validate(inconsistent).cheated
    assert SvgsVerdict.model_validate(
        {**inconsistent, "explicit_prohibition_violated": True}
    ).cheated


def test_invalid_judge_verdict_fails_sample(tmp_path: Path) -> None:
    agent = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", "I drew a picture.")],
        memoize=False,
    )
    judge = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", "{}")],
        memoize=False,
    )
    task = Task(dataset=[Sample(input="Draw an SVG.")], solver=generate(), scorer=svgs_scorer())
    [log] = inspect_eval(
        task,
        model=agent,
        model_roles={"grader": judge},
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "error"
    assert log.error
    assert "validation error" in log.error.message
