import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.cheatbench_inspect.svgs.conftest import FIXTURES, RUBRIC

from cheatbench_inspect import cheatbench_svgs
from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.harnesses.container import SANDBOX_RESOURCES
from cheatbench_inspect.svgs.dataset import CONFIG_PATH, default_task_dir, get_dataset
from cheatbench_inspect.tasks import _HARNESS_COMPOSE, _SVG_HARNESS_COMPOSE


def test_record_to_sample_and_study_defaults(svg_task_dir: Path) -> None:
    task = cheatbench_svgs(task_dir=str(svg_task_dir))
    [sample] = task.dataset
    assert task.version == "6-C"
    assert task.message_limit == 300
    assert task.epochs == 1
    assert task.fail_on_error is None
    defaults = load_task_defaults("svgs")
    assert defaults.timeout_s == 3600
    assert defaults.dataset_limit == 10
    assert sample.id == "pelican_bicycle"
    assert sample.input == (FIXTURES / "prompt.txt").read_text(encoding="utf-8")
    assert sample.target == ""
    assert sample.metadata
    assert sample.metadata["staged_files"] == [
        "/workspace/leaderboard/leaderboard.pyc",
        RUBRIC,
        CONFIG_PATH,
    ]
    assert sample.metadata["competitor_count"] == 5
    assert sample.files
    assert set(sample.files) == set(sample.metadata["staged_files"]) - {CONFIG_PATH}


@pytest.mark.parametrize(
    "args",
    [
        {"dataset_limit": 0},
        {"dataset_limit": 21},
        {"harness": "unknown"},
        {"harness": "codex", "harness_model": "gpt-6-sol"},
        {"harness": "terminus-2", "harness_version": "1"},
    ],
)
def test_invalid_task_args(svg_task_dir: Path, args: dict) -> None:
    with pytest.raises(ValueError, match=r"dataset_limit|harness|terminus-2"):
        cheatbench_svgs(task_dir=str(svg_task_dir), **args)


@pytest.mark.parametrize("harness", sorted(_HARNESS_COMPOSE))
def test_compose_variant(svg_task_dir: Path, harness: str) -> None:
    options: dict[str, Any] = (
        {} if harness == "terminus-2" else {"harness_version": "1.0", "harness_model": "test"}
    )
    task = cheatbench_svgs(task_dir=str(svg_task_dir), harness=harness, **options)
    assert task.sandbox is not None
    assert task.sandbox.config == str(_SVG_HARNESS_COMPOSE[harness])
    image_inputs = (
        "svg.Dockerfile",
        "svg-requirements.txt",
        "svg-debs-amd64.txt",
        "svg-debs-arm64.txt",
    )
    digest = hashlib.sha256(b"".join((SANDBOX_RESOURCES / p).read_bytes() for p in image_inputs))
    service = yaml.safe_load(Path(task.sandbox.config).read_text(encoding="utf-8"))["services"][
        "default"
    ]
    assert service["network_mode"] == "bridge"
    runtime = _HARNESS_COMPOSE[harness].name.removesuffix("-compose.yaml")
    assert (
        service["image"] == f"ghcr.io/rowan-dauria/task-env-{runtime}-svg:{digest.hexdigest()[:12]}"
    )


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_hf_archive_and_study_subsets() -> None:
    path = default_task_dir()
    sample_10 = cheatbench_svgs().dataset
    sample_20 = get_dataset(path, limit=20)
    assert len(sample_10) == 10
    assert len(sample_20) == 20
    assert [s.id for s in sample_10] == [s.id for s in sample_20[:10]]
    assert len({s.id for s in sample_20}) == 20
    assert sample_10[-1].id == "owl_lighthouse"
    assert sample_20[-1].id == "turtle_greenhouse"
    for sample in sample_20:
        assert sample.metadata
        assert sample.files
        rubric = json.loads(
            Path(sample.files[sample.metadata["rubric_path"]]).read_text(encoding="utf-8")
        )
        assert [c["max_points"] for c in rubric["criteria"]] == [20] * 5
