"""Load the published Creative Writings interviewer rows."""

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset

from cheatbench_inspect.image_task import render_prompt
from cheatbench_inspect.sandbox_task import fetch_task_dir

RESOURCES = Path(__file__).parent / "resources"
HF_REPO_ID = "rowan-dauria/cheatbench-creative-writings"
HF_REVISION = "624ebdb0b8088d10505318694d775657f4e1a4a1"
ARCHIVE_NAME = "creative_writings.tar.gz"
ARCHIVE_SHA256 = "ce3555ce1fc4340d1d7811b9d55475421829ef66ea91f23b08d1a49746e8f01e"


def default_task_dir() -> Path:
    """Download and verify the pinned interviewer archive."""
    return fetch_task_dir(
        HF_REPO_ID, ARCHIVE_NAME, HF_REVISION, ARCHIVE_SHA256, "creative_writings"
    )


def record_to_sample(record: dict[str, Any], environment_dir: Path) -> Sample:
    files = {
        f"/workspace/{name}": str(environment_dir / relative)
        for name, relative in record["files"].items()
    }
    files.update(
        {name: str(environment_dir / relative) for name, relative in record["files_abs"].items()}
    )
    template = (RESOURCES / "interviewer.txt").read_text(encoding="utf-8")
    return Sample(
        id=record["id"],
        input=render_prompt(template, record),
        files=files,
        metadata={
            "staged_files": sorted(files),
            "reference_path": record["meta"]["reference_path"],
            "topic": record["meta"]["topic"],
            "score_extensions": {
                arch: str(environment_dir / relative)
                for arch, relative in record["score_extensions"].items()
            },
        },
    )


def get_dataset(task_dir: Path, limit: int | None = None) -> Dataset:
    return json_dataset(
        str(task_dir / "data.jsonl"),
        sample_fields=lambda record: record_to_sample(record, task_dir / "environment"),
        limit=limit,
    )
