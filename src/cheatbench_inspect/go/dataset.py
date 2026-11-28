"""Load the published Go dataset."""

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset

from cheatbench_inspect.image_task import render_prompt
from cheatbench_inspect.sandbox_task import fetch_task_dir

RESOURCES = Path(__file__).parent / "resources"
HF_REPO_ID = "rowan-dauria/cheatbench-go"
HF_REVISION = "e4fd773bb27dd5f9afce5fc90072dc746fcc424a"
ARCHIVE_NAME = "go.tar.gz"
ARCHIVE_SHA256 = "a73845d7019f5dd7441deb4544ed18e5a0fdd36dd82094dd22b11decf3ae29ee"


def record_to_sample(record: dict[str, Any], *, environment_dir: Path) -> Sample:
    """Convert one built Go row into an Inspect Sample."""
    if record.get("tool_surface") != "go":
        raise ValueError("Invalid Go row: expected tool_surface='go'.")
    files = {
        f"/workspace/{dest}": str(environment_dir / source)
        for dest, source in (record.get("files") or {}).items()
    }
    return Sample(
        id=record["id"],
        input=render_prompt((RESOURCES / "instruction.txt").read_text(encoding="utf-8"), record),
        files=files,
        metadata={
            "staged_files": sorted(files),
            "fixture": record["fixture"],
            "go_tools": record["go_tools"],
            "tool_surface": record["tool_surface"],
        },
    )


def default_task_dir() -> Path:
    """Download, verify and unpack the published task data, once per archive."""
    return fetch_task_dir(HF_REPO_ID, ARCHIVE_NAME, HF_REVISION, ARCHIVE_SHA256, "go")


def get_dataset(task_dir: Path, *, limit: int | None = None) -> Dataset:
    """Load rows in the released builder's format."""
    return json_dataset(
        str(task_dir / "data.jsonl"),
        sample_fields=lambda record: record_to_sample(
            record, environment_dir=task_dir / "environment"
        ),
        limit=limit,
    )
