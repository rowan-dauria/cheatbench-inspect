"""Rebuild the release from the pinned original without publishing the site capture."""

import gzip
import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import ExitStack
from pathlib import Path
from urllib.request import urlopen

import zstandard

RESOURCES = Path(__file__).parent / "resources"
REVISION = "96e05a6d61faf33be788f75dd01998066045ad02"
UPSTREAM = (
    f"https://raw.githubusercontent.com/centerforaisafety/cheatbench/{REVISION}/tasks/openmath"
)
CAPTURE = "environment/site.tar.zst"
ZIP_SNAPSHOT = RESOURCES / "materials.tar.gz"
SITE_BUNDLE = "site.tar.xz"

# Only build_archive is replaced; build() and verify_archives() still check the
# original member bytes and build the captured-template posts and route index.
_BUILD_WITH_SNAPSHOT = """
import runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
main = runpy.run_path(sys.argv[1])["main"]
def original_archive(task_id, *_):
    path = main.__globals__["ARCHIVES_DIR"] / f"{task_id}.zip"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
main.__globals__["build_archive"] = original_archive
sys.argv = [sys.argv[1], "--variant", "nopw"]
raise SystemExit(main())
"""


def write_archive(destination: Path, root: Path, sources: list[Path]) -> None:
    """Write a deterministic gzip or xz tarball without host ownership or timestamps."""

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ""
        info.mode = 0o755 if info.isdir() else 0o644
        return info

    with ExitStack() as stack:
        stream = stack.enter_context(destination.open("wb"))
        if destination.suffix == ".xz":
            compressed = stack.enter_context(lzma.LZMAFile(stream, "wb", preset=3))
        else:
            compressed = stack.enter_context(
                gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="", compresslevel=3)
            )
        archive = stack.enter_context(
            tarfile.open(fileobj=compressed, mode="w|", format=tarfile.GNU_FORMAT)
        )
        for source in sources:
            archive.add(source, arcname=source.relative_to(root).as_posix(), filter=normalize)


def fetch_sources(source: Path, upstream_task: Path | None) -> None:
    checksums: dict[str, str] = json.loads((RESOURCES / "source_sha256.json").read_text())
    for name, expected in checksums.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if upstream_task is not None:
            shutil.copyfile(upstream_task / name, path)
        else:
            with urlopen(f"{UPSTREAM}/{name}", timeout=120) as response, path.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"{name} has sha256 {actual}, expected {expected}")


def build_task_dir(
    output: Path, *, upstream_task: Path | None = None, regenerate_materials: Path | None = None
) -> None:
    """Prepare data.jsonl, row responses and a portable site bundle in an empty directory."""
    with tempfile.TemporaryDirectory(prefix="capture-") as temporary:
        source = Path(temporary)
        fetch_sources(source, upstream_task)
        with (
            (source / CAPTURE).open("rb") as stream,
            zstandard.ZstdDecompressor().stream_reader(stream) as decompressed,
            tarfile.open(fileobj=decompressed, mode="r|") as capture,
        ):
            # The pinned archive contains only trusted upstream paths. The
            # data filter also rejects unsafe links and out-of-tree writes.
            capture.extractall(source, filter="data")
        site = source / "environment/site"
        (source / "rh_mirror").rename(site)
        generated = source / "environment/nopw"
        zip_dir = generated / "files/archives"
        env = {**os.environ, "TZ": "UTC", "PYTHONIOENCODING": "utf-8", "ZIPOPT": ""}
        if regenerate_materials is not None:
            command = [sys.executable, str(source / "build.py"), "--variant", "nopw"]
        else:
            with tarfile.open(ZIP_SNAPSHOT) as materials:
                materials.extractall(zip_dir, filter="data")
            command = [sys.executable, "-c", _BUILD_WITH_SNAPSHOT, str(source / "build.py")]
        try:
            subprocess.run(
                command, env=env, check=True, capture_output=True, text=True, umask=0o022
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Upstream build failed: {error.stderr or error.stdout}") from error
        if regenerate_materials is not None:
            write_archive(regenerate_materials, zip_dir, sorted(zip_dir.glob("*.zip")))

        output.mkdir(parents=True, exist_ok=True)
        # Keep the capture's bodies, headers and search index and omit its
        # crawler and pregenerated upstream TLS keys.
        write_archive(
            output / SITE_BUNDLE,
            site,
            [site / name for name in ("cache", "manifest.json", "pages")],
        )
        for name in ("files", "overrides.json", "overrides.index.json", "data.jsonl"):
            src = generated / name
            if src.is_dir():
                shutil.copytree(src, output / name)
            else:
                shutil.copyfile(src, output / name)
