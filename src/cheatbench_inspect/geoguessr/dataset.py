"""Load the published GeoGuessr dataset."""

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH

from cheatbench_inspect.image_task import image_record_to_sample
from cheatbench_inspect.image_task import render_prompt as render_prompt
from cheatbench_inspect.sandbox_task import fetch_task_dir

# The original builder's output, pinned by commit and checksum.
# scripts/geoguessr_dataset.py builds and optionally uploads it.
HF_REPO_ID = "rowan-dauria/cheatbench-geoguessr"
HF_REVISION = "15cf7e0831e63cf32c2bb6236866c492a2037654"
ARCHIVE_NAME = "geoguessr.tar.gz"
ARCHIVE_SHA256 = "d1174f3fbc067111ce8cb95f4c2fdb76ac499320041faa55011b1c1b65add6c6"

# The original single variant has 20 frozen photographs and --repeat 1.
DEFAULT_DATASET_LIMIT = 20

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
        "geoguessr",
        cache_root=INSPECT_EVALS_CACHE_PATH,
    )


def get_dataset(task_dir: Path, *, limit: int | None = None) -> Dataset:
    """Load the GeoGuessr single-variant rows from a prepared task directory."""
    return json_dataset(
        str(task_dir / "data.jsonl"),
        sample_fields=lambda record: record_to_sample(
            record, environment_dir=task_dir / "environment"
        ),
        limit=limit,
    )
