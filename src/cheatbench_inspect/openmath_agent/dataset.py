"""Load original OpenMath Agent rows and their per-episode repository manifests."""

import hashlib
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH

from cheatbench_inspect.image_task import render_prompt
from cheatbench_inspect.openmath_agent.build import (
    RESOURCES,
    SOURCE_HASHES,
    UPSTREAM,
    build_task_dir,
)


def default_task_dir(cache_root: Path = INSPECT_EVALS_CACHE_PATH) -> Path:
    """Publish the complete build atomically for concurrent eval workers."""
    digest = hashlib.sha256(UPSTREAM.encode() + SOURCE_HASHES.read_bytes()).hexdigest()
    task_dir = cache_root / "cheatbench_inspect" / "openmath_agent" / digest[:12]
    if (task_dir / "data.jsonl").is_file():
        return task_dir

    task_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=task_dir.parent) as temporary:
        output = Path(temporary) / "data"
        build_task_dir(output)
        try:
            output.rename(task_dir)
        except OSError:
            if not (task_dir / "data.jsonl").is_file():
                raise
    return task_dir


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or not path.parts:
        raise ValueError(f"Expected a relative workspace path, got {value!r}")
    return path


def record_to_sample(record: dict[str, Any], template: str) -> Sample:
    return Sample(
        id=record["id"],
        input=render_prompt(template, record),
        metadata={
            "staged_files": [],
            "repository_files": record["files"],
            "repository_mtimes": record.get("file_mtimes", {}),
            "repository_modes": record.get("file_modes", {}),
            "deliverable_files": record["deliverable_files"],
        },
    )


def get_dataset(task_dir: Path, *, limit: int | None = None) -> Dataset:
    template = (RESOURCES / "instruction.txt").read_text(encoding="utf-8")
    return json_dataset(
        str(task_dir / "data.jsonl"),
        sample_fields=lambda record: record_to_sample(record, template),
        limit=limit,
    )
