# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/tasks/knowledge_work/build.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

"""Build the selected GDPval rows locally from checksum-verified Hub files."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH
from inspect_evals_utils.huggingface import hf_hub_download

from cheatbench_inspect.image_task import render_prompt

RESOURCES = Path(__file__).parent / "resources"
SELECTION = json.loads((RESOURCES / "samples20.json").read_text())
HF_REPO_ID: str = SELECTION["dataset"]
HF_REVISION: str = SELECTION["revision"]
TASK_IDS: tuple[str, ...] = tuple(row["task_id"] for row in SELECTION["tasks"])
PARQUET = "data/train-00000-of-00001.parquet"
CHECKSUMS_PATH = RESOURCES / "sha256.json"
REVIEW_ROOT = "/data/cache/artifacts"


def verified_download(filename: str, checksums: dict[str, str]) -> Path:
    expected = checksums[filename]
    path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID, filename=filename, repo_type="dataset", revision=HF_REVISION
        )
    )
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"{filename} has sha256 {digest}, expected {expected}")
    return path


def write_row(output: Path, item: dict[str, Any], checksums: dict[str, str]) -> dict[str, Any]:
    task_id = item["task_id"]
    if len(item["deliverable_files"]) != 1:
        raise ValueError(f"{task_id}: expected one deliverable")
    row_id = task_id[:8]
    environment = output / "environment"
    directory = environment / "files" / row_id
    directory.mkdir(parents=True)
    files: dict[str, str] = {}
    absolute: dict[str, str] = {}
    for field, category, prefix, destinations in (
        ("reference_files", "reference_files", "inputs", files),
        ("deliverable_files", "gold_deliverable_files", REVIEW_ROOT, absolute),
    ):
        for remote in item[field]:
            basename = Path(remote).name
            key = f"{prefix}/{basename}"
            if key in destinations:
                raise ValueError(f"{task_id}: duplicate input basename {basename}")
            target = directory / category / basename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(verified_download(remote, checksums), target)
            destinations[key] = str(target.relative_to(environment))
    brief = item["prompt"].replace("reference_files/", "inputs/")
    (directory / "brief.md").write_text(brief + "\n", encoding="utf-8")
    shutil.copyfile(RESOURCES / "workspace_readme.txt", directory / "README.md")
    for name in ("brief.md", "README.md"):
        files[name] = str((directory / name).relative_to(environment))
    return {
        "id": row_id,
        "files": files,
        "files_abs": absolute,
        "reference_destination": next(iter(absolute)),
        "meta": {
            "task_id": task_id,
            "sector": item["sector"],
            "occupation": item["occupation"],
            "gold_name": Path(item["deliverable_files"][0]).name,
        },
    }


def default_task_dir(cache_root: Path = INSPECT_EVALS_CACHE_PATH) -> Path:
    """Download and materialize the release once in the Inspect cache."""
    manifest = CHECKSUMS_PATH.read_bytes()
    checksums = json.loads(manifest)
    digest = hashlib.sha256(
        HF_REVISION.encode()
        + (RESOURCES / "samples20.json").read_bytes()
        + (RESOURCES / "workspace_readme.txt").read_bytes()
        + manifest
    ).hexdigest()
    task_dir = cache_root / "cheatbench_inspect" / "knowledge_work" / digest[:12]
    if (task_dir / "data.jsonl").is_file():
        return task_dir

    from pyarrow.parquet import read_table

    task_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=task_dir.parent) as temporary:
        output = Path(temporary)
        table = read_table(
            verified_download(PARQUET, checksums),
            filters=[("task_id", "in", list(TASK_IDS))],
        )
        items = {item["task_id"]: item for item in table.to_pylist()}
        rows = [write_row(output, items[task_id], checksums) for task_id in TASK_IDS]
        (output / "data.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        try:
            output.rename(task_dir)
        except OSError:
            # Another worker may have published the same completed build.
            if not (task_dir / "data.jsonl").is_file():
                raise
    return task_dir


def record_to_sample(record: dict[str, Any], *, environment_dir: Path) -> Sample:
    files = {
        f"/workspace/{dest}": str(environment_dir / source)
        for dest, source in record["files"].items()
    }
    files.update(
        {dest: str(environment_dir / source) for dest, source in record["files_abs"].items()}
    )
    prompt = render_prompt((RESOURCES / "instruction.txt").read_text(encoding="utf-8"), record)
    return Sample(
        id=record["id"],
        input=prompt,
        files=files,
        metadata={
            "staged_files": sorted(files),
            "reference_destination": record["reference_destination"],
            "meta": record.get("meta", {}),
        },
    )


def get_dataset(task_dir: Path, *, limit: int | None = None) -> Dataset:
    """Load an original-format Knowledge Work build."""
    return json_dataset(
        str(task_dir / "data.jsonl"),
        sample_fields=lambda record: record_to_sample(
            record, environment_dir=task_dir / "environment"
        ),
        limit=limit,
    )
