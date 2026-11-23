"""Load the original OpenMath rows and select the response files for each sample."""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH

from cheatbench_inspect.image_task import render_prompt
from cheatbench_inspect.openmath.build import RESOURCES, REVISION, ZIP_SNAPSHOT, build_task_dir


def default_task_dir(cache_root: Path = INSPECT_EVALS_CACHE_PATH) -> Path:
    """Cache a complete build keyed by its pinned inputs."""
    digest = hashlib.sha256(
        REVISION.encode()
        + (RESOURCES / "source_sha256.json").read_bytes()
        + ZIP_SNAPSHOT.read_bytes()
    ).hexdigest()
    task_dir = cache_root / "cheatbench_inspect" / "openmath" / digest[:12]
    if (task_dir / "data.jsonl").is_file():
        return task_dir

    task_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=task_dir.parent) as temporary:
        output = Path(temporary) / "data"
        build_task_dir(output)
        try:
            output.rename(task_dir)
        except OSError:
            # A concurrent worker may have published the same complete build.
            if not (task_dir / "data.jsonl").is_file():
                raise
    return task_dir


def row_responses(task_dir: Path, row_id: str) -> dict[str, bytes]:
    """Select exactly the original post, redirect, ZIP and index entry for a row."""
    if not row_id or Path(row_id).name != row_id or row_id in {".", ".."}:
        raise ValueError(f"Invalid row ID {row_id!r}")
    entries = json.loads((task_dir / "overrides.index.json").read_text())
    [entry] = [item for item in entries if item["task_id"] == row_id]
    post = entry["path"]
    overrides = json.loads((task_dir / "overrides.json").read_text())
    post_file = f"files/posts/{row_id}.html"
    archive_file = f"files/archives/{row_id}.zip"
    [archive_path] = [
        url for url, spec in overrides.items() if spec.get("body_file") == archive_file
    ]
    if overrides[post].get("body_file") != post_file:
        raise ValueError(f"Post override does not belong to {row_id}")
    selected = {url: overrides[url] for url in (post, post.rstrip("/"), archive_path)}
    return {
        "overrides.json": json.dumps(selected).encode(),
        "overrides.index.json": json.dumps([entry]).encode(),
        post_file: (task_dir / post_file).read_bytes(),
        archive_file: (task_dir / archive_file).read_bytes(),
    }


def record_to_sample(record: dict[str, Any], template: str) -> Sample:
    return Sample(
        id=record["id"],
        input=render_prompt(template, record),
        metadata={"staged_files": []},
    )


def get_dataset(task_dir: Path, *, limit: int | None = None) -> Dataset:
    template = (RESOURCES / "instruction.txt").read_text(encoding="utf-8")
    return json_dataset(
        str(task_dir / "data.jsonl"),
        sample_fields=lambda record: record_to_sample(record, template),
        limit=limit,
    )
