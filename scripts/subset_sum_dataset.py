"""Build the Subset Sum data with the original's builder and pack it for Hugging Face.

The original does not ship this task's data: its build.py generates data.jsonl
and the environment/ tree, offline and deterministically. This script runs that
builder, pinned by commit and checksum, and packs its output into a tarball. The task downloads that tarball from a Hugging Face dataset repo when no
task_dir is given.

    uv run python scripts/subset_sum_dataset.py OUT_DIR
    HF_TOKEN=... uv run python scripts/subset_sum_dataset.py OUT_DIR --upload REPO_ID

Upload prints the Hugging Face commit to pin as HF_REVISION, and the archive's
sha256 to pin as ARCHIVE_SHA256, both in cheatbench_inspect/subset_sum/dataset.py.
"""

import argparse
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

UPSTREAM_COMMIT = "96e05a6d61faf33be788f75dd01998066045ad02"
BUILDER_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/cheatbench/"
    f"{UPSTREAM_COMMIT}/tasks/subset_sum/build.py"
)
BUILDER_SHA256 = "e1cf45f9967752a0c41202593f35be6f2c6e8df4505aa794390e02cf047b4c0a"
ARCHIVE_NAME = "subset_sum.tar.gz"

DATASET_CARD = f"""---
license: mit
pretty_name: CheatBench Subset Sum (generated)
---

# CheatBench Subset Sum, generated data

The output of the [CheatBench](https://github.com/centerforaisafety/cheatbench)
Subset Sum builder at commit `{UPSTREAM_COMMIT}`, run with its default
arguments (questions Q4-Q13): `data.jsonl` and the `environment/` tree staged
into each episode's sandbox, packed as `{ARCHIVE_NAME}`.

The files include the task's planted answer keys, which are part of the
benchmark. The builder is deterministic, so anyone can regenerate these files
from the public source. The fixture includes the planted checkout verbatim.

Packed by `scripts/subset_sum_dataset.py` in
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
    (work / "build.py").write_bytes(source)
    assets = work / "assets"
    assets.mkdir()
    for name, expected in {
        "fixture.zip": "85389d994e8bb0a2ae4e780a82b6e30253d2548ca3ba6e8e3a7ad461f243bd8e",
        "fixture.zip.sha256": "28d72a2806665a27a6f78fd7c7ac97e979a922f594f2d7e0c181a2484886e041",
    }.items():
        with urllib.request.urlopen(
            BUILDER_URL.rsplit("/", 1)[0] + "/assets/" + name, timeout=60
        ) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch for {name}")
        (assets / name).write_bytes(data)
    if (
        hashlib.sha256((assets / "fixture.zip").read_bytes()).hexdigest()
        != (assets / "fixture.zip.sha256").read_text().split()[0]
    ):
        raise RuntimeError("fixture checksum file mismatch")
    subprocess.run([sys.executable, "-B", str(work / "build.py")], cwd=work, check=True)


def write_fixture(work: Path, rows: list[int], out: Path) -> None:
    """Copy selected rows and precisely the files they stage."""
    records = [json.loads(line) for line in (work / "data.jsonl").read_text().splitlines()]
    selected = [row for row in records if row["meta"]["question"] in rows]
    if len(selected) != len(set(rows)):
        raise ValueError("Some requested fixture rows were not generated")
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        (fixture / "data.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
        for row in selected:
            for name in set(row["files"].values()) | set(row["files_abs"].values()):
                source = work / "environment" / name
                target = fixture / "environment" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        out.parent.mkdir(parents=True, exist_ok=True)
        pack(fixture, out)


def pack(work: Path, archive: Path) -> str:
    """Pack data.jsonl and environment/ and return the archive's sha256.

    Entries are sorted and their owners and timestamps fixed. The planted
    checkout is unpacked from the pinned fixture, so rebuilds are byte-identical.
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
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
    ):
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
        commit_message=f"Subset Sum data from cheatbench {UPSTREAM_COMMIT[:7]}",
    )
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack the Subset Sum data and optionally upload it."
    )
    parser.add_argument("out_dir", type=Path, help="Directory to write the archive to.")
    parser.add_argument("--upload", metavar="REPO_ID", help="Hugging Face dataset repo.")
    parser.add_argument("--fixture-rows", nargs="+", type=int)
    parser.add_argument("--fixture-out", type=Path)
    args = parser.parse_args()
    if bool(args.fixture_rows) != bool(args.fixture_out):
        parser.error("--fixture-rows and --fixture-out must be provided together")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / ARCHIVE_NAME
    with tempfile.TemporaryDirectory() as work:
        run_builder(fetch_builder(), Path(work))
        digest = pack(Path(work), archive)
        if args.fixture_out:
            write_fixture(Path(work), args.fixture_rows, args.fixture_out)
    print(f"{archive}: sha256 {digest}")
    if args.upload:
        print(f"Uploaded to {args.upload} at revision {upload(args.upload, archive)}")


if __name__ == "__main__":
    main()
