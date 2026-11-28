import json
from pathlib import Path

import pytest

import cheatbench_inspect
from cheatbench_inspect import tasks


def test_export_and_default_task(tmp_path: Path) -> None:
    row = json.loads((Path(__file__).parent / "fixtures/q4.json").read_text())
    rows = [{**row, "id": str(i)} for i in range(12)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert cheatbench_inspect.cheatbench_subset_sum is tasks.cheatbench_subset_sum
    assert "cheatbench_subset_sum" in cheatbench_inspect.__all__
    task = tasks.cheatbench_subset_sum(task_dir=str(tmp_path))
    assert task.epochs == 1
    assert task.message_limit == 2 * 150
    assert len(task.dataset) == 10
    assert len(tasks.cheatbench_subset_sum(task_dir=str(tmp_path), dataset_limit=2).dataset) == 2


@pytest.mark.parametrize(
    "harness", ["terminus-2", "claude-code", "codex", "gemini-cli", "kimi-code"]
)
def test_compose_selection(tmp_path: Path, harness: str) -> None:
    (tmp_path / "data.jsonl").write_text(
        (Path(__file__).parent / "fixtures/q4.json").read_text().replace("\n", "") + "\n"
    )
    version = None if harness == "terminus-2" else "1"
    model = None if harness == "terminus-2" else "model"
    task = tasks.cheatbench_subset_sum(
        task_dir=str(tmp_path), harness=harness, harness_version=version, harness_model=model
    )
    assert task.sandbox is not None
    assert task.sandbox.config == str(tasks._HARNESS_COMPOSE[harness])


@pytest.mark.parametrize(
    "options",
    [
        {"harness": "unknown"},
        {"harness": "codex"},
        {"harness": "claude-code"},
        {"harness_version": "1"},
        {"harness_model": "model"},
        {"dataset_limit": 0},
    ],
)
def test_invalid_arguments(options: dict) -> None:
    with pytest.raises(ValueError, match=r"harness|terminus-2|dataset_limit"):
        tasks.cheatbench_subset_sum(**options)
