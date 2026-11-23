"""Load the published Subset Sum dataset."""

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH

from cheatbench_inspect.sandbox_task import fetch_task_dir

HF_REPO_ID = "rowan-dauria/cheatbench-subset-sum"
HF_REVISION = "586b789eec1da8fa3a18ce71cbe4f3e56a0211c9"
ARCHIVE_NAME = "subset_sum.tar.gz"
ARCHIVE_SHA256 = "a8c0195eb315ebaea68ab3e0cbc24f82129bd22405c617e055d30568899a4f9e"


def record_to_sample(record: dict[str, Any], *, environment_dir: Path) -> Sample:
    files = {
        f"/workspace/{dest}": str(environment_dir / source)
        for dest, source in record["files"].items()
    }
    files.update(
        {dest: str(environment_dir / source) for dest, source in record["files_abs"].items()}
    )
    metadata = {
        "numbers": record["numbers"],
        "target": record["target"],
        "staged_files": sorted(files),
    }
    return Sample(id=record["id"], input=record["input"], metadata=metadata, files=files)


def default_task_dir() -> Path:
    """Download, verify and unpack the published task data, once per archive."""
    return fetch_task_dir(
        HF_REPO_ID,
        ARCHIVE_NAME,
        HF_REVISION,
        ARCHIVE_SHA256,
        "subset_sum",
        cache_root=INSPECT_EVALS_CACHE_PATH,
    )


def get_dataset(task_dir: Path, *, limit: int | None = None) -> Dataset:
    return json_dataset(
        str(task_dir / "data.jsonl"),
        sample_fields=lambda record: record_to_sample(
            record, environment_dir=task_dir / "environment"
        ),
        limit=limit,
    )
