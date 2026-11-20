import json
from pathlib import Path
from typing import Any

from inspect_ai import eval
from inspect_ai.model import ChatMessage, ModelOutput, get_model
from pytest_inspect_evals.assertions import assert_eval_success

from cheatbench_inspect import cheatbench_sycophancy


def test_sycophancy_defaults_to_published_sampling_count() -> None:
    task = cheatbench_sycophancy()
    assert task.epochs == 100
    assert len(task.dataset) == 28


def test_sycophancy_generates_one_reply_per_condition(tmp_path: Path) -> None:
    expected_samples = {sample.id: sample for sample in cheatbench_sycophancy().dataset}
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(model="mockllm/model", content="0.4") for _ in range(28)
        ],
    )

    [log] = eval(
        tasks=cheatbench_sycophancy(),
        model=model,
        epochs=1,
        score=False,
        log_dir=str(tmp_path),
        display="none",
    )

    assert_eval_success(log, require_scorer=False, expected_sample_count=28)
    assert log.samples is not None
    assert {sample.id for sample in log.samples} == set(expected_samples)
    for sample in log.samples:
        expected = expected_samples[sample.id]
        assert sample.metadata == expected.metadata
        # Original interaction: tasks/sycophancy/run_chat.py:335-338.
        assert [message.role for message in sample.messages] == ["user", "assistant"]
        assert sample.messages[0].text == expected.input
        assert sample.messages[1].text == "0.4"
        assert sample.output.completion == "0.4"
        assert not sample.scores


def test_sycophancy_end_to_end_matches_original_shift(tmp_path: Path) -> None:
    # The "typical" fixture case: 28 samples x 5 epochs of judge estimates, including
    # UNKNOWN, out-of-range and blank replies, with the original score.py's result.
    [case] = [
        c
        for c in json.loads((Path(__file__).parent / "fixtures" / "shift_cases.json").read_text())
        if c["name"] == "typical"
    ]
    prompts = {sample.input: str(sample.id) for sample in cheatbench_sycophancy().dataset}
    replies: dict[str, list[str]] = {}
    for record in case["records"]:
        estimate = record["estimate"]
        replies.setdefault(record["id"], []).append("" if estimate is None else str(estimate))
    queues = {sample_id: iter(r) for sample_id, r in replies.items()}

    def evaluated(messages: list[ChatMessage], *_: Any) -> ModelOutput:
        reply = next(queues[prompts[messages[-1].text]])
        return ModelOutput.from_content(model="mockllm/model", content=reply)

    def grader(messages: list[ChatMessage], *_: Any) -> ModelOutput:
        # Echo the number the evaluated model replied with, as a verdict.
        reply = messages[-1].text.split("\nASSISTANT: ")[1].split("\n</trajectory>")[0]
        verdict = {
            "cheated": False,
            "estimate": float(reply),
            "committed": True,
            "summary": "",
            "explicit_prohibition_violated": False,
            "explicit_prohibition_evidence": "",
        }
        return ModelOutput.from_content(model="mockllm/model", content=json.dumps(verdict))

    [log] = eval(
        tasks=cheatbench_sycophancy(),
        model=get_model("mockllm/model", custom_outputs=evaluated, memoize=False),
        model_roles={"grader": get_model("mockllm/model", custom_outputs=grader, memoize=False)},
        epochs=5,
        log_dir=str(tmp_path),
        display="none",
    )

    assert_eval_success(log, expected_sample_count=28 * 5)
    assert log.results is not None
    [scores] = log.results.scores
    metrics = {name: m.value for name, m in scores.metrics.items()}
    expected = case["expected"]
    assert metrics["shift"] == expected["percent"]
    assert metrics["forecast"] == expected["forecast_percent"]
    assert metrics["bubble"] == expected["bubble_percent"]
    assert [metrics["ci95_lower"], metrics["ci95_upper"]] == expected["ci95"]
