"""Build the pinned Synthetic Puzzles dataset in an offline Python 3.11 container.

    uv run python scripts/synthetic_puzzles_dataset.py OUT_DIR
    uv run python scripts/synthetic_puzzles_dataset.py OUT_DIR --upload-existing REPO_ID

The second command uploads the completed archive and prints its revision.
"""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from geoguessr_dataset import (
    BASE_IMAGE,
    EXIFTOOL_DEB_VERSION,
    PILLOW_VERSION,
    PYYAML_VERSION,
    RENDER_PROMPT_BOOTSTRAP,
    UPSTREAM_COMMIT,
    ensure_builder_image,
    fetch_verified,
    pack,
    run_builder_in_container,
    write_fixture,
)

TASK = "tasks/synthetic_puzzles"
ARCHIVE_NAME = "synthetic_puzzles.tar.gz"
UPSTREAM_FILES = {
    f"{TASK}/build.py": "cbaf1a5736013c02b00ed6c9f557317a93e1a4a76b4b42665cb24519e4520503",
    f"{TASK}/fixture.py": "795c52db327e60ca36f37176758d63dacb09f918e645219d28dc4272b593fe9e",
    f"{TASK}/task.yaml": "38b6f94006d3c3a0a75d8e9c1984732d9632a093ef786479aeec9c4d85044a42",
    f"{TASK}/prompts/instruction.md": "230e10d08187bc9dc8c4185631b7060628b2b793f24e01f5fbd537bead33157a",
    f"{TASK}/round_images.json": "953e222700c9da375963f7cd58ea756755f9a5ed4fbd0a5be21a679f77d2cf51",
    "core/trial.py": "c1fca24f3ac41029aa605c5518a811895eddbca252abf2d23fa50ad5a19fcf67",
    "LICENSE": "05424bacbf4ec79bf8c2c8a0bf0eaf0f7b1b74379df933cb45b20b4b31248a22",
}
PHOTO_SHA256 = {
    "131": "90f2e45c5f58499bb9f5e6edf6b82171d3341ecc6010d14aa44b0ffcddbf3f1a",
    "275": "43166269ab5bc548ce386ef19c01dbaa1fe41957fb64c6bff9a75c539b5c5f05",
    "300": "c134aad629c59d400a0414b37bdb8e30f419d6ff5d22c53827360fde761e5ad3",
    "383": "195c6c264102adcc5e549d785377ad7c74d0e88acee2b272f6286463f8c857a8",
    "468": "b3199c48794585a130cbc77b40c2d35f78f223fc2f409bb671c8d6e1deadcaf8",
    "730": "f26fe8b1f394d59dab39ca3cdfa79617f126c529ada557b09d1349da8ceaf5bc",
    "733": "a86ad99c1ec93e7787530ce71842b9ed39775822cad462029031539e1e8abf47",
    "739": "7ec6c9072c019068c5f526961021667c089264466984729bab4461804b94938e",
    "855": "73f6b523361aea889419a938dd18653e6e2102b92463aba2452b975312935efd",
    "954": "e0cab1d1316fa3bd379fd34a20cb0f1dda416fcb8f5d11e5cc91f308f67cab77",
}

BUILDER_DOCKERFILE = f"""FROM {BASE_IMAGE}
COPY --from=ghcr.io/astral-sh/uv@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git=1:2.47.3-0+deb13u1 libimage-exiftool-perl={EXIFTOOL_DEB_VERSION} \
    && rm -rf /var/lib/apt/lists/*
RUN uv venv /opt/build && uv pip install --python /opt/build/bin/python --no-cache \
    numpy==2.2.6 scipy==1.15.3 pillow=={PILLOW_VERSION} pyyaml=={PYYAML_VERSION}
ENV PATH="/opt/build/bin:$PATH"
"""

# Reuse the original render_prompt-only bootstrap. Normalize the Git index's
# cached filesystem stat fields, which otherwise depend on the build checkout.
PUZZLE_ENTRYPOINT = (
    RENDER_PROMPT_BOOTSTRAP
    + f"""
import subprocess

task = Path("/build/{TASK}")
sys.argv = [str(task / "build.py")]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as exc:
    if exc.code != 0:
        raise
git_dir = task / "environment/files/workspace/generator/.git"
(git_dir / "index").unlink()
subprocess.run(["git", f"--git-dir={{git_dir}}", "read-tree", "HEAD"], check=True)
"""
)


def fetch_inputs(repo: Path, photos_dir: Path | None) -> None:
    for relative, digest in UPSTREAM_FILES.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fetch_verified(relative, digest))

    cache = repo / TASK / ".cache"
    cache.mkdir()
    for photo_id, expected in PHOTO_SHA256.items():
        name = f"picsum_{photo_id}_4200x2800.jpg"
        if photos_dir is None:
            request = urllib.request.Request(
                f"https://picsum.photos/id/{photo_id}/4200/2800",
                headers={"User-Agent": "reward-hacking-eval/0.1 (research)"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                photo = response.read()
        else:
            photo = (photos_dir / name).read_bytes()
        digest = hashlib.sha256(photo).hexdigest()
        if digest != expected:
            raise ValueError(f"{name} has sha256 {digest}, expected {expected}")
        (cache / name).write_bytes(photo)


def build(out_dir: Path, photos_dir: Path | None, fixture_out: Path | None) -> None:
    image = ensure_builder_image(BUILDER_DOCKERFILE, "puzzle-builder")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        repo = root / "repo"
        staged = root / "staged"
        staged.mkdir()
        fetch_inputs(repo, photos_dir)
        run_builder_in_container(
            repo, image, entrypoint=PUZZLE_ENTRYPOINT, interpreter="/opt/build/bin/python3.11"
        )
        task = repo / TASK
        shutil.copyfile(task / "data.jsonl", staged / "data.jsonl")
        shutil.copyfile(task / "round_images.json", staged / "round_images.json")
        shutil.copyfile(repo / "LICENSE", staged / "LICENSE")
        shutil.move(str(task / "environment"), staged / "environment")
        archive = out_dir / ARCHIVE_NAME
        digest = pack(staged, archive, root_files=("data.jsonl", "round_images.json", "LICENSE"))
        if fixture_out:
            write_fixture(
                staged,
                ["puzzle_img_01"],
                fixture_out,
                root_files=("data.jsonl",),
                exclude=("files/workspace/puzzle/img_01.jpg",),
            )
        print(f"{archive}: sha256 {digest}")


def upload(repo_id: str, archive: Path) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    with tarfile.open(archive) as tar:
        photo_manifest = tar.extractfile("round_images.json")
        assert photo_manifest is not None
        photos = json.load(photo_manifest)
    credits = "\n".join(f"- [{photo['author']}]({photo['url']})" for photo in photos)
    card = f"""---
license: other
pretty_name: CheatBench Synthetic Puzzles (generated)
---

# CheatBench Synthetic Puzzles

Twenty generated puzzles from [CheatBench](https://github.com/centerforaisafety/cheatbench)
at commit `{UPSTREAM_COMMIT}`, built for the Inspect AI port.
The archive includes the rows, environment, source-photo attribution and MIT license
for the benchmark code. Puzzle JPEGs incorporate photographs from Picsum under the
[Unsplash License](https://unsplash.com/license). Each photograph has two cuts.

{credits}
"""
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    commit = api.create_commit(
        repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(ARCHIVE_NAME, str(archive)),
            CommitOperationAdd("README.md", card.encode()),
        ],
        commit_message=f"Synthetic Puzzles data from CheatBench {UPSTREAM_COMMIT[:7]}",
    )
    print(
        f"{repo_id} revision {commit.oid}, sha256 {hashlib.sha256(archive.read_bytes()).hexdigest()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--photos-dir", type=Path, help="Reuse already downloaded pinned photos.")
    parser.add_argument("--fixture-out", type=Path)
    parser.add_argument("--upload-existing", metavar="REPO_ID")
    args = parser.parse_args()
    if args.upload_existing:
        upload(args.upload_existing, args.out_dir / ARCHIVE_NAME)
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        build(args.out_dir, args.photos_dir, args.fixture_out)


if __name__ == "__main__":
    main()
