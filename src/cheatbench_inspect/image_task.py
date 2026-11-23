"""Dataset rows shared by image tasks with local compiled checkers."""

from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample


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


def image_record_to_sample(
    record: dict[str, Any], *, environment_dir: Path, template_path: Path
) -> Sample:
    """Stage the original row's workspace and absolute paths for a local-checker episode."""
    files = {
        f"/workspace/{dest}": str(environment_dir / source)
        for dest, source in record["files"].items()
    }
    files.update(
        {dest: str(environment_dir / source) for dest, source in record["files_abs"].items()}
    )
    template = template_path.read_text(encoding="utf-8")
    prompt = render_prompt(template, record)
    metadata: dict[str, Any] = {
        "image": record["input"],
        "staged_files": sorted(files),
    }
    if "meta" in record:
        metadata["meta"] = record["meta"]
    if "file_mtimes" in record:
        metadata["file_mtimes"] = record["file_mtimes"]
    return Sample(id=record["id"], input=prompt, metadata=metadata, files=files)
