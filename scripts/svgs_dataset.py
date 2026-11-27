"""Build and publish the workshop data from the pinned original using Python 3.11.

Run with `uv run --no-project --python 3.11 --with huggingface-hub python
scripts/svgs_dataset.py /home/agent/cheatbench OUT_DIR`. Use `--upload REPO_ID`
with a Hugging Face write token to publish the archive and dataset card.
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from subset_sum_dataset import pack

UPSTREAM_COMMIT = "96e05a6d61faf33be788f75dd01998066045ad02"
ARCHIVE_NAME = "svgs.tar.gz"

DATASET_CARD = f"""---
license: mit
pretty_name: CheatBench SVG Competition (workshop)
---

The 20 workshop rows produced by the [CheatBench](https://github.com/centerforaisafety/cheatbench)
SVG builder at `{UPSTREAM_COMMIT}`, including the planted rubrics and the Python 3.11
submitter. The first ten rows form the default study subset. Some cohorts used all twenty.

The archive includes the upstream MIT license. No reference images or third-party artworks are staged.
"""


def prepare(upstream: Path, out: Path, *, fixture: bool = False) -> None:
    """Run the original builder off-checkout and collect the files Inspect stages."""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The dataset requires Python 3.11 bytecode.")
    source = subprocess.check_output(
        ["git", "-C", str(upstream), "archive", UPSTREAM_COMMIT, "tasks/svgs", "LICENSE"]
    )
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(source)) as archive:
            archive.extractall(work, filter="data")
        task = work / "tasks/svgs"
        subprocess.run(
            [sys.executable, str(task / "build.py")],
            check=True,
            env={**os.environ, "SOURCE_DATE_EPOCH": "0"},
        )
        built = task / "environment/workshop"
        out.mkdir(parents=True, exist_ok=True)
        environment = out / "environment"
        environment.mkdir()
        shutil.copyfile(work / "LICENSE", environment / "LICENSE")
        shutil.copytree(built / "shared", environment / "shared")
        if fixture:
            first, *_ = (built / "data.jsonl").read_bytes().splitlines(keepends=True)
            (out / "data.jsonl").write_bytes(first)
            shutil.copytree(built / "rows/pelican_bicycle", environment / "rows/pelican_bicycle")
        else:
            shutil.copyfile(built / "data.jsonl", out / "data.jsonl")
            shutil.copytree(built / "rows", environment / "rows")


def upload(repo_id: str, archive: Path) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    return api.create_commit(
        repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(ARCHIVE_NAME, str(archive)),
            CommitOperationAdd("README.md", DATASET_CARD.encode()),
        ],
        commit_message=f"SVG workshop data from CheatBench {UPSTREAM_COMMIT[:7]}",
    ).oid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--upload", metavar="REPO_ID")
    parser.add_argument("--fixture", action="store_true", help="Pack only the first workshop row.")
    args = parser.parse_args()
    if args.fixture and args.upload:
        parser.error("--fixture cannot be published as the full dataset")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / (ARCHIVE_NAME if not args.fixture else "workshop_fixture.tar.gz")
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        prepare(args.upstream, work, fixture=args.fixture)
        print(f"{archive} SHA-256: {pack(work, archive)}")
    if args.upload:
        print(f"{args.upload} revision: {upload(args.upload, archive)}")


if __name__ == "__main__":
    main()
