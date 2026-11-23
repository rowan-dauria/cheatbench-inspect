"""Load the published GeoGuessr dataset."""

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH

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


def render_prompt(template: str, record: dict[str, Any]) -> str:
    """Fill a prompt template from a record.

    Verbatim port of CheatBench's `core/trial.py:render_prompt`:
    https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/trial.py#L415-L440
    """
    fields = {"input": record.get("input") or "", **(record.get("vars") or {})}
    out = []
    for template_line in template.split("\n"):
        tokens = [k for k in fields if "{{" + k + "}}" in template_line]
        if tokens and all(not str(fields[k]).strip() for k in tokens):
            continue
        line = template_line
        for k, v in fields.items():
            line = line.replace("{{" + k + "}}", str(v))
        out.append(line)
    text = "\n".join(out)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"


def record_to_sample(record: dict[str, Any], *, environment_dir: Path) -> Sample:
    """Convert one GeoGuessr data.jsonl row into an Inspect sample."""
    files = {
        f"/workspace/{dest}": str(environment_dir / source)
        for dest, source in record["files"].items()
    }
    files.update(
        {dest: str(environment_dir / source) for dest, source in record["files_abs"].items()}
    )
    template = INSTRUCTION_TEMPLATE_PATH.read_text(encoding="utf-8")
    prompt = render_prompt(template, record)
    metadata: dict[str, Any] = {
        "image": record["input"],
        "staged_files": sorted(files),
    }
    if "meta" in record:
        metadata["meta"] = record["meta"]
    return Sample(id=record["id"], input=prompt, metadata=metadata, files=files)


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
