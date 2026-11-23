"""Load the published Prime Factorization dataset."""

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH
from inspect_evals_utils.huggingface import hf_hub_download

from cheatbench_inspect.sandbox_task import fetch_task_dir

# The original builder's output, hosted as a Hugging Face dataset and pinned by
# commit and checksum. scripts/prime_factorization_dataset.py publishes it.
HF_REPO_ID = "rowan-dauria/cheatbench-prime-factorization"
HF_REVISION = "113b091eabac9023eea3a47086f7a16ee08acb0b"
ARCHIVE_NAME = "prime_factorization.tar.gz"
ARCHIVE_SHA256 = "cd985aa28ef9ff1127e697e559de2eafc0b0305cb8c3c16f49e2ce40b29e878c"

# The original README's reference commands use --limit 10.
DEFAULT_DATASET_LIMIT = 10


def record_to_sample(record: dict[str, Any], *, environment_dir: Path) -> Sample:
    files = {
        f"/workspace/{destination}": str(environment_dir / source)
        for destination, source in record["files"].items()
    }
    for destination, source in record["files_abs"].items():
        files[destination] = str(environment_dir / source)

    metadata: dict[str, Any] = {"number": int(record["number"]), "staged_files": sorted(files)}
    return Sample(id=record["id"], input=record["input"], metadata=metadata, files=files)


def default_task_dir() -> Path:
    """Download, verify and unpack the published task data, once per archive.

    The original generates this data with its build.py. The archive holds that
    builder's output at the pinned commit, packed by
    scripts/prime_factorization_dataset.py.
    """
    return fetch_task_dir(
        HF_REPO_ID,
        ARCHIVE_NAME,
        HF_REVISION,
        ARCHIVE_SHA256,
        "prime_factorization",
        cache_root=INSPECT_EVALS_CACHE_PATH,
        download=hf_hub_download,
    )


def get_dataset(task_dir: Path, *, limit: int | None = None) -> Dataset:
    data_path = task_dir / "data.jsonl"
    return json_dataset(
        str(data_path),
        sample_fields=lambda record: record_to_sample(
            record, environment_dir=task_dir / "environment"
        ),
        limit=limit,
    )
