"""Load the published Synthetic Puzzles dataset."""

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH

from cheatbench_inspect.image_task import image_record_to_sample
from cheatbench_inspect.sandbox_task import fetch_task_dir

# The original builder's output, pinned by commit and checksum.
# scripts/synthetic_puzzles_dataset.py builds and optionally uploads it.
HF_REPO_ID = "rowan-dauria/cheatbench-synthetic-puzzles"
HF_REVISION = "632714ac77ec57ece1e6d5be799158bd4afa0d2b"
ARCHIVE_NAME = "synthetic_puzzles.tar.gz"
ARCHIVE_SHA256 = "12b6e355684131e8b9492dcf628c9ee71febe499a4ab0a8ac9f14fc25310db5c"


INSTRUCTION_TEMPLATE_PATH = Path(__file__).parent / "resources/instruction.txt"


def record_to_sample(record: dict[str, Any], *, environment_dir: Path) -> Sample:
    return image_record_to_sample(
        record, environment_dir=environment_dir, template_path=INSTRUCTION_TEMPLATE_PATH
    )


def default_task_dir() -> Path:
    """Download, verify and unpack the published task data, once per archive."""
    return fetch_task_dir(
        HF_REPO_ID,
        ARCHIVE_NAME,
        HF_REVISION,
        ARCHIVE_SHA256,
        "synthetic_puzzles",
        cache_root=INSPECT_EVALS_CACHE_PATH,
    )


def get_dataset(task_dir: Path, *, limit: int | None = None) -> Dataset:
    """Load the Synthetic Puzzles rows from a prepared task directory."""
    return json_dataset(
        str(task_dir / "data.jsonl"),
        sample_fields=lambda record: record_to_sample(
            record, environment_dir=task_dir / "environment"
        ),
        limit=limit,
    )
