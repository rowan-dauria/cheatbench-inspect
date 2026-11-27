"""Load the pinned workshop briefs, rubrics and compiled submitter."""

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset

from cheatbench_inspect.image_task import render_prompt
from cheatbench_inspect.sandbox_task import fetch_task_dir

RESOURCES = Path(__file__).with_name("resources")
HF_REPO_ID = "rowan-dauria/cheatbench-svgs"
HF_REVISION = "9baa0522577894e31449c76ab3fb9a68d57dec1b"
ARCHIVE_NAME = "svgs.tar.gz"
ARCHIVE_SHA256 = "ca25918f3e9020aff33f6f1c021def82d35eb985e8b19f72a9e2bf299965a09c"
CONFIG_PATH = "/workspace/leaderboard/leaderboard_config.json"


def default_task_dir() -> Path:
    return fetch_task_dir(HF_REPO_ID, ARCHIVE_NAME, HF_REVISION, ARCHIVE_SHA256, "svgs")


def record_to_sample(record: dict[str, Any], environment_dir: Path, template: str) -> Sample:
    ((rubric_path, rubric_source),) = record["files_abs"].items()
    files = {
        "/workspace/leaderboard/leaderboard.pyc": str(
            environment_dir / record["files"]["leaderboard/leaderboard.pyc"]
        ),
        rubric_path: str(environment_dir / rubric_source),
    }
    return Sample(
        id=record["id"],
        input=render_prompt(template, record),
        files=files,
        metadata={
            **record["meta"],
            "rubric_path": rubric_path,
            "staged_files": [*files, CONFIG_PATH],
        },
    )


def get_dataset(task_dir: Path, *, limit: int | None = None) -> Dataset:
    template = (RESOURCES / "instruction.txt").read_text(encoding="utf-8")
    return json_dataset(
        str(task_dir / "data.jsonl"),
        sample_fields=lambda row: record_to_sample(row, task_dir / "environment", template),
        limit=limit,
    )
