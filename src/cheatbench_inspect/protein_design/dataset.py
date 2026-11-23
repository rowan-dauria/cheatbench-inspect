"""Load the published Protein Design dataset."""

from pathlib import Path, PurePosixPath
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset

from cheatbench_inspect.image_task import render_prompt
from cheatbench_inspect.sandbox_task import fetch_task_dir

RESOURCES = Path(__file__).parent / "resources"
HF_REPO_ID = "rowan-dauria/cheatbench-protein-design"
HF_REVISION = "856324fbc1d922e5daefa4566e882b66fd0997a7"
ARCHIVE_NAME = "protein_design.tar.gz"
ARCHIVE_SHA256 = "5234f5eb1afe0830445d5d3fe81d4fb2101d62fa8de65f40a343f616e9f79ec7"


def record_to_sample(record: dict[str, Any], *, environment_dir: Path) -> Sample:
    files = {dest: str(environment_dir / source) for dest, source in record["files_abs"].items()}
    file_mtimes: dict[str, str] = record["file_mtimes"]
    # The original also ages the directories created during staging, using
    # the newest staged timestamp under each directory.
    directory_mtimes: dict[str, str] = {}
    for path, stamp in file_mtimes.items():
        for parent in PurePosixPath(path).parents:
            if str(parent) in {"/tmp", "/home/user", "/"}:
                break
            directory = str(parent)
            directory_mtimes[directory] = max(directory_mtimes.get(directory, stamp), stamp)
    return Sample(
        id=record["id"],
        input=render_prompt((RESOURCES / "instruction.txt").read_text(encoding="utf-8"), record),
        files=files,
        metadata={
            "staged_files": sorted(files),
            "file_mtimes": {**file_mtimes, **directory_mtimes},
            "work_dir": record["vars"]["work_dir"],
            "payload_file": record["payload_file"],
            "pull_log": record["pull_log"],
        },
    )


def default_task_dir() -> Path:
    """Download, verify and unpack the published task data, once per archive."""
    return fetch_task_dir(HF_REPO_ID, ARCHIVE_NAME, HF_REVISION, ARCHIVE_SHA256, "protein_design")


def get_dataset(task_dir: Path, *, limit: int | None = None) -> Dataset:
    """Load rows in the released builder's format."""
    return json_dataset(
        str(task_dir / "data.jsonl"),
        sample_fields=lambda record: record_to_sample(
            record, environment_dir=task_dir / "environment"
        ),
        limit=limit,
    )
