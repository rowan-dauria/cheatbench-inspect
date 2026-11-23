"""Download, verify and cache the published Subset Sum data."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from cheatbench_inspect import sandbox_task, tasks
from cheatbench_inspect.subset_sum import dataset as subset_sum

ROW = {
    "id": "subsetsum_n60d10_4",
    "input": "Factor.",
    "numbers": [1, 2],
    "target": 3,
    "live_log_dir": "/logs",
    "cache_dir": "/cache",
    "files": {},
    "files_abs": {},
}


def make_archive(path: Path) -> str:
    """Write a small archive shaped like the published one and return its sha256."""
    data = (json.dumps(ROW) + "\n").encode()
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("data.jsonl")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        environment = tarfile.TarInfo("environment")
        environment.type = tarfile.DIRTYPE
        tar.addfile(environment)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    archive = tmp_path / "download" / subset_sum.ARCHIVE_NAME
    archive.parent.mkdir()
    monkeypatch.setattr(subset_sum, "ARCHIVE_SHA256", make_archive(archive))
    monkeypatch.setattr(subset_sum, "INSPECT_EVALS_CACHE_PATH", tmp_path / "cache")
    return archive


def test_default_task_dir_downloads_verifies_and_unpacks(published: Path) -> None:
    with patch.object(sandbox_task, "hf_hub_download", return_value=str(published)) as download:
        task_dir = subset_sum.default_task_dir()

    download.assert_called_once_with(
        repo_id=subset_sum.HF_REPO_ID,
        filename=subset_sum.ARCHIVE_NAME,
        repo_type="dataset",
        revision=subset_sum.HF_REVISION,
    )
    assert task_dir.parent.name == "subset_sum"
    assert task_dir.name == subset_sum.ARCHIVE_SHA256[:12]
    assert json.loads((task_dir / "data.jsonl").read_text()) == ROW
    assert (task_dir / "environment").is_dir()


def test_default_task_dir_reuses_the_cache(published: Path) -> None:
    with patch.object(sandbox_task, "hf_hub_download", return_value=str(published)):
        first = subset_sum.default_task_dir()
    with patch.object(sandbox_task, "hf_hub_download") as download:
        assert subset_sum.default_task_dir() == first
    download.assert_not_called()


def test_default_task_dir_rejects_a_changed_archive(
    published: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subset_sum, "ARCHIVE_SHA256", "0" * 64)
    with (
        patch.object(sandbox_task, "hf_hub_download", return_value=str(published)),
        pytest.raises(ValueError, match="sha256"),
    ):
        subset_sum.default_task_dir()

    assert not (subset_sum.INSPECT_EVALS_CACHE_PATH / "cheatbench_inspect").exists()


def test_task_without_task_dir_uses_the_published_data(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text(json.dumps(ROW) + "\n")
    with patch.object(subset_sum, "default_task_dir", return_value=tmp_path) as default:
        task = tasks.cheatbench_subset_sum()

    default.assert_called_once_with()
    assert [sample.id for sample in task.dataset] == ["subsetsum_n60d10_4"]


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_published_archive_loads_q4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subset_sum, "INSPECT_EVALS_CACHE_PATH", tmp_path)
    task_dir = subset_sum.default_task_dir()

    dataset = subset_sum.get_dataset(task_dir, limit=1)
    assert [sample.id for sample in dataset] == ["subsetsum_n60d10_4"]
    sample = dataset[0]
    assert sample.metadata is not None
    assert sample.metadata["target"] == 163721718194
    assert sample.files
    assert "/workspace/instance.txt" in sample.files
    assert all(Path(source).is_file() for source in sample.files.values())
