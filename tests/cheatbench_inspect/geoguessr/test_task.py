"""Tests for cheatbench_geoguessr task registration, defaults, validation, and compose."""

import shutil
from pathlib import Path

import pytest
import yaml

import cheatbench_inspect
from cheatbench_inspect import tasks
from cheatbench_inspect.geoguessr import dataset as geoguessr
from cheatbench_inspect.geoguessr.geoguessr import MAX_TURNS

FIXTURES = Path(__file__).parent / "fixtures"


def test_export_and_default_task(tmp_path: Path, row0_task_dir: Path) -> None:
    shutil.copyfile(FIXTURES / "rows.jsonl", tmp_path / "data.jsonl")
    shutil.copyfile(row0_task_dir / "manifest.json", tmp_path / "manifest.json")

    assert cheatbench_inspect.cheatbench_geoguessr is tasks.cheatbench_geoguessr
    assert "cheatbench_geoguessr" in cheatbench_inspect.__all__

    task = tasks.cheatbench_geoguessr(task_dir=str(tmp_path))
    assert task.epochs == 1
    assert task.message_limit == 2 * MAX_TURNS
    assert len(task.dataset) == geoguessr.DEFAULT_DATASET_LIMIT == 20
    assert len(tasks.cheatbench_geoguessr(task_dir=str(tmp_path), dataset_limit=2).dataset) == 2


@pytest.mark.parametrize("harness", ["terminus-2", "claude-code", "codex"])
def test_compose_selection(row0_task_dir: Path, harness: str) -> None:
    version = None if harness == "terminus-2" else "1"
    model = None if harness == "terminus-2" else "model"
    task = tasks.cheatbench_geoguessr(
        task_dir=str(row0_task_dir), harness=harness, harness_version=version, harness_model=model
    )
    assert task.sandbox is not None
    offline_path = tasks._OFFLINE_HARNESS_COMPOSE[harness]
    open_path = tasks._HARNESS_COMPOSE[harness]
    assert task.sandbox.config == str(offline_path)
    assert offline_path.name.endswith("compose.yaml")

    offline_service = yaml.safe_load(offline_path.read_text(encoding="utf-8"))["services"][
        "default"
    ]
    open_service = yaml.safe_load(open_path.read_text(encoding="utf-8"))["services"]["default"]
    assert offline_service["network_mode"] == "none"
    assert "network_mode" not in open_service
    assert offline_service["image"] == open_service["image"]
    assert offline_service["build"] == open_service["build"]
    assert "extends" not in offline_service
    assert "platform" not in offline_service


@pytest.mark.parametrize(
    "options",
    [
        {"harness": "unknown"},
        {"harness": "codex"},
        {"harness": "codex", "harness_version": "0.154.0"},
        {"harness": "codex", "harness_model": "gpt-5.6-luna"},
        {"harness": "claude-code"},
        {"harness": "claude-code", "harness_version": "2.1.270"},
        {"harness": "claude-code", "harness_model": "claude-sonnet-5"},
        {"harness_version": "1"},
        {"harness_model": "model"},
        {"dataset_limit": 0},
    ],
)
def test_invalid_arguments(options: dict) -> None:
    with pytest.raises(ValueError, match=r"harness|terminus-2|dataset_limit"):
        tasks.cheatbench_geoguessr(**options)
