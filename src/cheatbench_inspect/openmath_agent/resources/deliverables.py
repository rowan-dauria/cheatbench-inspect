# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/runner_support.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

"""Export saved work from the episode as its unprivileged user."""

import os
import sys

DELIVERABLE_EXTS = (
    ".xlsx",
    ".xls",
    ".xlsm",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".pdf",
    ".csv",
    ".svg",
)
_DELIVERABLE_MAX_BYTES = 64 * 1024 * 1024


def collect_deliverables(root: str, skip_dirs: tuple, include_files: tuple = ()) -> list:
    """Export office files and exact task-requested outputs for the host."""
    import base64
    import stat

    requested = {
        p
        for p in include_files
        if isinstance(p, str)
        and p
        and (not os.path.isabs(p))
        and (not any(part in ("", ".", "..") for part in p.split("/")))
    }
    out, total = ([], 0)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        top = rel_dir.split(os.sep)[0]
        if top in skip_dirs:
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            relative = os.path.relpath(os.path.join(dirpath, name), root)
            if (
                os.path.splitext(name)[1].lower() not in DELIVERABLE_EXTS
                and relative not in requested
            ):
                continue
            q = os.path.join(dirpath, name)
            try:
                info = os.lstat(q)
                if not stat.S_ISREG(info.st_mode):
                    continue
                remaining = _DELIVERABLE_MAX_BYTES - total
                if info.st_size > remaining:
                    continue
                with open(q, "rb") as f:
                    data = f.read(remaining + 1)
            except OSError:
                continue
            if total + len(data) > _DELIVERABLE_MAX_BYTES:
                sys.stderr.write(f"deliverable export cap hit; skipping {q}")
                continue
            total += len(data)
            out.append({"name": os.path.relpath(q, root), "b64": base64.b64encode(data).decode()})
    return out
