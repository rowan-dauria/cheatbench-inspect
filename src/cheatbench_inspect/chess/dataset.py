"""Load the published Chess dataset."""

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset

from cheatbench_inspect.image_task import render_prompt
from cheatbench_inspect.sandbox_task import fetch_task_dir

RESOURCES = Path(__file__).parent / "resources"
HF_REPO_ID = "rowan-dauria/cheatbench-chess"
HF_REVISION = "85448c8fcad8a2e183986c7b70d878aad9359a5c"
ARCHIVE_NAME = "chess.tar.gz"
ARCHIVE_SHA256 = "238dd03fadc40f6514795d582baec112b5e28f48a1917fddf967cc42fdf52604"


def record_to_sample(record: dict[str, Any], *, environment_dir: Path) -> Sample:
    """Convert one built Chess row into an Inspect Sample."""
    if record.get("variant") == "none" and record.get("tool_surface") != "chess":
        raise ValueError("Stale five-tool chess rows: rebuild with scripts/chess_dataset.py.")
    files = {
        f"/workspace/{dest}": str(environment_dir / source)
        for dest, source in record["files"].items()
    }
    return Sample(
        id=record["id"],
        input=render_prompt((RESOURCES / "instruction.txt").read_text(encoding="utf-8"), record),
        files=files,
        metadata={
            "staged_files": sorted(files),
            "fen": record["fen"],
            "solution_san": record["solution_san"],
            "round_number": int(record.get("round_number", 4)),
            "game_number": int(record.get("game_number", 7)),
            "variant": record.get("variant", "none"),
            "tool_surface": record.get("tool_surface", "chess"),
        },
    )


def default_task_dir() -> Path:
    """Download, verify and unpack the published task data, once per archive."""
    return fetch_task_dir(HF_REPO_ID, ARCHIVE_NAME, HF_REVISION, ARCHIVE_SHA256, "chess")


def get_dataset(task_dir: Path, *, limit: int | None = None) -> Dataset:
    """Load rows in the released builder's format."""
    return json_dataset(
        str(task_dir / "data.jsonl"),
        sample_fields=lambda record: record_to_sample(
            record, environment_dir=task_dir / "environment"
        ),
        limit=limit,
    )
