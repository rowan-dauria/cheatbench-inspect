"""Build the Prime Factorization data with the original's builder and pack it for Hugging Face.

The original does not ship this task's data: its build.py generates data.jsonl
and the environment/ tree, offline and deterministically. This script runs that
builder, pinned by commit and checksum, and packs its output into a tarball. The task downloads that tarball from a Hugging Face dataset repo when no
task_dir is given.

    uv run python scripts/prime_factorization_dataset.py OUT_DIR
    HF_TOKEN=... uv run python scripts/prime_factorization_dataset.py OUT_DIR --upload REPO_ID

Upload prints the Hugging Face commit to pin as HF_REVISION, and the archive's
sha256 to pin as ARCHIVE_SHA256, both in cheatbench_inspect/prime_factorization/dataset.py.
"""

import argparse
import gzip
import hashlib
import io
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

UPSTREAM_COMMIT = "96e05a6d61faf33be788f75dd01998066045ad02"
BUILDER_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/cheatbench/"
    f"{UPSTREAM_COMMIT}/tasks/prime_factorization/build.py"
)
BUILDER_SHA256 = "54cd657bf5b8d24f392907e37197accc13241546ff71ca44572214d051c09acb"
ARCHIVE_NAME = "prime_factorization.tar.gz"

DATASET_CARD = f"""---
license: mit
pretty_name: CheatBench Prime Factorization (generated)
---

# CheatBench Prime Factorization, generated data

The output of the [CheatBench](https://github.com/centerforaisafety/cheatbench)
Prime Factorization builder at commit `{UPSTREAM_COMMIT}`, run with its default
arguments (questions Q4-Q23): `data.jsonl` and the `environment/` tree staged
into each episode's sandbox, packed as `{ARCHIVE_NAME}`.

The files include the task's planted answer keys, which are part of the
benchmark. The builder is deterministic, so anyone can regenerate these files
from the public source. A rebuild differs only in the planted checkout's
`.git/index`, where git caches file timestamps.

Packed by `scripts/prime_factorization_dataset.py` in
[cheatbench-inspect](https://github.com/rowan-dauria/cheatbench-inspect), the
Inspect AI port that downloads it.

CheatBench is copyright (c) 2026 centerforaisafety and Long Phan, MIT licensed.
"""


def fetch_builder() -> bytes:
    with urllib.request.urlopen(BUILDER_URL, timeout=60) as response:
        source = response.read()
    digest = hashlib.sha256(source).hexdigest()
    if digest != BUILDER_SHA256:
        raise RuntimeError(f"build.py has sha256 {digest}, expected {BUILDER_SHA256}")
    return source


def run_builder(source: bytes, work: Path) -> None:
    # The builder writes data.jsonl and environment/ next to itself.
    builder = work / "build.py"
    builder.write_bytes(source)
    subprocess.run([sys.executable, "-B", str(builder)], cwd=work, check=True)


def pack(work: Path, archive: Path) -> str:
    """Pack data.jsonl and environment/ and return the archive's sha256.

    The task reads only file contents, so entries are sorted and their owners and
    timestamps fixed. Rebuilding still changes one file: the planted checkout's
    .git/index, where git caches the build machine's file timestamps and inode
    numbers. Every other file comes out byte-identical, which is why the task pins
    the published archive by its sha256.
    """
    paths = [work / "data.jsonl", work / "environment"]
    paths += sorted((work / "environment").rglob("*"))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in paths:
            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(work)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if info.isdir() else 0o644
            if info.isfile():
                with path.open("rb") as file:
                    tar.addfile(info, file)
            else:
                tar.addfile(info)
    with archive.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(buffer.getvalue())
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def upload(repo_id: str, archive: Path) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    commit = api.create_commit(
        repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(ARCHIVE_NAME, str(archive)),
            CommitOperationAdd("README.md", DATASET_CARD.encode()),
        ],
        commit_message=f"Prime Factorization data from cheatbench {UPSTREAM_COMMIT[:7]}",
    )
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack the Prime Factorization data and optionally upload it."
    )
    parser.add_argument("out_dir", type=Path, help="Directory to write the archive to.")
    parser.add_argument("--upload", metavar="REPO_ID", help="Hugging Face dataset repo.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / ARCHIVE_NAME
    with tempfile.TemporaryDirectory() as work:
        run_builder(fetch_builder(), Path(work))
        digest = pack(Path(work), archive)
    print(f"{archive}: sha256 {digest}")
    if args.upload:
        print(f"Uploaded to {args.upload} at revision {upload(args.upload, archive)}")


if __name__ == "__main__":
    main()
