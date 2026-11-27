"""Rebuild the research repositories from the checksum-pinned release."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cheatbench_inspect.openmath.build import REVISION, fetch_pinned_files

RESOURCES = Path(__file__).parent / "resources"
UPSTREAM = (
    f"https://raw.githubusercontent.com/centerforaisafety/cheatbench/{REVISION}"
    "/tasks/openmath_agent"
)
SOURCE_HASHES = RESOURCES / "source_sha256.json"

# The task's original builder verifies each reference download against sources.json.
# TLS is set up in each episode, so skip generating private keys in the build.
_BUILD = """
import runpy, sys

from pathlib import Path
namespace = runpy.run_path(sys.argv[1])
build = namespace["build"]
build.__globals__["certificates"] = lambda site: None
output = Path(sys.argv[2])
rows = build(output)
for row in rows:
    repo = output / "environment/files" / row["id"]
    (repo / ".git/index").unlink()
    build.__globals__["git"](repo, "read-tree", "HEAD")
"""


def build_task_dir(
    output: Path, *, upstream_task: Path | None = None, assets: Path | None = None
) -> None:
    """Build original rows and repository files into an empty directory."""
    with tempfile.TemporaryDirectory(prefix="research-") as temporary:
        source = Path(temporary)
        fetch_pinned_files(source, UPSTREAM, json.loads(SOURCE_HASHES.read_text()), upstream_task)
        if assets is not None:
            shutil.copytree(assets, source / ".cache")
        try:
            subprocess.run(
                [sys.executable, "-I", "-c", _BUILD, str(source / "build.py"), str(output)],
                env={**os.environ, "TZ": "UTC", "PYTHONIOENCODING": "utf-8"},
                check=True,
                capture_output=True,
                text=True,
                umask=0o022,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Upstream build failed: {error.stderr or error.stdout}") from error
