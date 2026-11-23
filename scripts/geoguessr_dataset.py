"""Build the GeoGuessr data with the original's builder in Docker and pack it for Hugging Face.

The original does not ship this task's generated environment: its build.py verifies
the 20 frozen JPEGs, generates each row's GPX waypoint file, compiles the Python 3.11
checker (`submit.pyc`), runs metadata/isolation/scoring self-checks, and writes
`environment/single/data.jsonl`. Because build.py writes to `/tmp/gps` during its
self-checks, this script runs it exclusively inside an isolated Docker container with
no network access after image construction.

    uv run python scripts/geoguessr_dataset.py OUT_DIR
    HF_TOKEN=... uv run python scripts/geoguessr_dataset.py OUT_DIR --upload REPO_ID

Upload prints the Hugging Face commit to pin as HF_REVISION, and the archive's
sha256 to pin as ARCHIVE_SHA256, both in cheatbench_inspect/geoguessr/dataset.py.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

UPSTREAM_COMMIT = "96e05a6d61faf33be788f75dd01998066045ad02"
RAW_BASE = f"https://raw.githubusercontent.com/centerforaisafety/cheatbench/{UPSTREAM_COMMIT}"

UPSTREAM_FILES: dict[str, str] = {
    "tasks/geoguessr/build.py": "9c08742c8d7ac807131a51c17ad235c80029fef5926e2840f56ed9cf37dbdc14",
    "tasks/geoguessr/task.yaml": "ced4f1417f00a3e384996a45fc1ea4ac441c8b5f71500e8cf85fcd69569d05e8",
    "tasks/geoguessr/prompts/instruction.md": (
        "652b9cffb6548f625c576d372895a0e0b1faa20f67d4c9615e473c5bbab884a3"
    ),
    "tasks/geoguessr/assets/manifest.json": (
        "fd71f7745a1de69a661510151985c732110b2925e13270891e44b420793898e7"
    ),
    "core/trial.py": "c1fca24f3ac41029aa605c5518a811895eddbca252abf2d23fa50ad5a19fcf67",
}

ARCHIVE_NAME = "geoguessr.tar.gz"

# Match the sandbox Dockerfile's pinned Python 3.11 base digest and the original's
# pinned Pillow (pyproject.toml:37, Dockerfile:54) and PyYAML (Dockerfile:56) versions.
# Debian trixie provides exiftool via libimage-exiftool-perl.
BASE_IMAGE = (
    "python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535"
)
EXIFTOOL_DEB_VERSION = "13.25+dfsg-1"
PILLOW_VERSION = "12.3.0"
PYYAML_VERSION = "6.0.3"

BUILDER_DOCKERFILE = f"""FROM {BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \\
        libimage-exiftool-perl={EXIFTOOL_DEB_VERSION} \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \\
        pillow=={PILLOW_VERSION} \\
        pyyaml=={PYYAML_VERSION}
"""

# Run the original tasks/geoguessr/build.py verbatim inside the build container.
# build.py imports only `render_prompt` from `core.trial` (build.py:38); core/trial.py
# has top-level imports for the rest of the original harness (pydantic, etc.), so we
# extract the verbatim `render_prompt` AST node from the pinned core/trial.py source
# and register it as `core.trial` before invoking build.py.
RENDER_PROMPT_BOOTSTRAP = """
import ast
import runpy
import sys
import types
from pathlib import Path

trial_path = Path("/build/core/trial.py")
tree = ast.parse(trial_path.read_text(encoding="utf-8"), filename=str(trial_path))
fn = next(
    (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_prompt"),
    None,
)
if fn is None:
    raise RuntimeError("render_prompt not found in core/trial.py")

mod_ast = ast.Module(body=[fn], type_ignores=[])
ast.fix_missing_locations(mod_ast)
ns = {"Path": Path}
exec(compile(mod_ast, str(trial_path), "exec"), ns)

core_pkg = types.ModuleType("core")
trial_mod = types.ModuleType("core.trial")
trial_mod.render_prompt = ns["render_prompt"]
core_pkg.trial = trial_mod
sys.modules["core"] = core_pkg
sys.modules["core.trial"] = trial_mod

"""

CONTAINER_ENTRYPOINT = (
    RENDER_PROMPT_BOOTSTRAP
    + """
sys.argv = ["/build/tasks/geoguessr/build.py", "--variant", "single"]
runpy.run_path("/build/tasks/geoguessr/build.py", run_name="__main__")
"""
)

DATASET_CARD = f"""---
license: mit
pretty_name: CheatBench GeoGuessr (generated)
---

# CheatBench GeoGuessr, generated data

The output of the [CheatBench](https://github.com/centerforaisafety/cheatbench)
GeoGuessr builder at commit `{UPSTREAM_COMMIT}`, run for the default `single`
variant (20 rows): `data.jsonl`, the evaluator `manifest.json` (kept outside
`environment/` and never staged to the agent), and the `environment/` tree staged
into each episode's sandbox, packed as `{ARCHIVE_NAME}`.

The files include the task's planted GPX reference coordinates, which are part of
the benchmark. The builder runs inside an isolated Python 3.11 container with
`SOURCE_DATE_EPOCH=0` and `PYTHONHASHSEED=0` so `submit.pyc` uses PEP 552
hash-based invalidation and rebuilds are byte-identical.

Packed by `scripts/geoguessr_dataset.py` in
[cheatbench-inspect](https://github.com/rowan-dauria/cheatbench-inspect), the
Inspect AI port that downloads it.

CheatBench is copyright (c) 2026 centerforaisafety and Long Phan, MIT licensed.
"""


def fetch_verified(rel_path: str, expected_sha256: str) -> bytes:
    url = f"{RAW_BASE}/{rel_path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"{rel_path} has sha256 {digest}, expected {expected_sha256}")
    return data


def fetch_inputs(repo_root: Path) -> None:
    """Download every pinned build input from upstream GitHub into repo_root."""
    for rel_path, expected in UPSTREAM_FILES.items():
        target = repo_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fetch_verified(rel_path, expected))

    manifest = json.loads(
        (repo_root / "tasks/geoguessr/assets/manifest.json").read_text(encoding="utf-8")
    )
    images_dir = repo_root / "tasks/geoguessr/assets/images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for photo in manifest["photos"]:
        rel_image = f"tasks/geoguessr/assets/{photo['image']}"
        data = fetch_verified(rel_image, photo["sha256"])
        (repo_root / rel_image).write_bytes(data)


def ensure_builder_image(
    dockerfile: str = BUILDER_DOCKERFILE, repository: str = "cheatbench-geoguessr-builder"
) -> str:
    """Build (or reuse) the pinned Python 3.11 + exiftool builder Docker image."""
    tag = hashlib.sha256(dockerfile.encode()).hexdigest()[:12]
    image = f"{repository}:{tag}"
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    if inspect.returncode == 0:
        return image
    with tempfile.TemporaryDirectory() as ctx:
        (Path(ctx) / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        subprocess.run(["docker", "build", "-t", image, ctx], check=True)
    return image


def run_builder_in_container(
    repo_root: Path,
    image: str,
    *,
    entrypoint: str = CONTAINER_ENTRYPOINT,
    interpreter: str = "python3.11",
) -> None:
    """Run build.py inside the builder container with networking disabled."""
    # SOURCE_DATE_EPOCH=0 makes Python 3.11's py_compile emit PEP 552 checked-hash
    # .pyc headers instead of wall-clock mtime headers; PYTHONHASHSEED=0 keeps
    # bytecode marshalling deterministic across runs.
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "SOURCE_DATE_EPOCH=0",
            "-e",
            "PYTHONHASHSEED=0",
            "-e",
            "RH_PY311=/usr/local/bin/python3.11",
            "-e",
            "RH_EXIFTOOL=/usr/bin/exiftool",
            "-v",
            f"{repo_root}:/build",
            image,
            interpreter,
            "-B",
            "-c",
            entrypoint,
        ],
        check=True,
    )


def stage_archive_tree(repo_root: Path, staged: Path) -> None:
    """Assemble the archive directory layout from build.py's output."""
    single_env = repo_root / "tasks/geoguessr/environment/single"
    shutil.copyfile(single_env / "data.jsonl", staged / "data.jsonl")
    shutil.copyfile(repo_root / "tasks/geoguessr/assets/manifest.json", staged / "manifest.json")
    shutil.copytree(single_env / "files", staged / "environment/files")


def pack(
    work: Path, archive: Path, *, root_files: tuple[str, ...] = ("data.jsonl", "manifest.json")
) -> str:
    """Pack data.jsonl, manifest.json and environment/ reproducibly and return sha256."""
    paths = [*(work / file for file in root_files), work / "environment"]
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


def write_fixture(
    staged: Path,
    row_ids: list[str],
    out: Path,
    *,
    root_files: tuple[str, ...] = ("data.jsonl", "manifest.json"),
    exclude: tuple[str, ...] = (),
) -> str:
    """Pack selected rows, manifest.json and precisely the files they stage."""
    records = [
        json.loads(line)
        for line in (staged / "data.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in records if row["id"] in set(row_ids)]
    if len(selected) != len(set(row_ids)):
        found = {row["id"] for row in selected}
        missing = sorted(set(row_ids) - found)
        raise ValueError(f"Requested fixture row(s) not found: {missing}")
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        (fixture / "data.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in selected),
            encoding="utf-8",
        )
        for file in root_files:
            if file != "data.jsonl":
                shutil.copyfile(staged / file, fixture / file)
        for row in selected:
            for rel in set(row["files"].values()) | set(row["files_abs"].values()):
                if rel in exclude:
                    continue
                source = staged / "environment" / rel
                target = fixture / "environment" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        out.parent.mkdir(parents=True, exist_ok=True)
        return pack(fixture, out, root_files=root_files)


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
        commit_message=f"GeoGuessr data from cheatbench {UPSTREAM_COMMIT[:7]}",
    )
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack the GeoGuessr data in Docker and optionally upload it."
    )
    parser.add_argument("out_dir", type=Path, help="Directory to write the archive to.")
    parser.add_argument("--upload", metavar="REPO_ID", help="Hugging Face dataset repo.")
    parser.add_argument(
        "--fixture-rows",
        nargs="+",
        metavar="ROW_ID",
        help="Row ID(s) to pack into a small test fixture archive.",
    )
    parser.add_argument("--fixture-out", type=Path, help="Output path for the test fixture.")
    args = parser.parse_args()
    if bool(args.fixture_rows) != bool(args.fixture_out):
        parser.error("--fixture-rows and --fixture-out must be provided together")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / ARCHIVE_NAME
    image = ensure_builder_image()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_root = tmp_path / "repo"
        staged = tmp_path / "staged"
        repo_root.mkdir()
        staged.mkdir()
        fetch_inputs(repo_root)
        run_builder_in_container(repo_root, image)
        stage_archive_tree(repo_root, staged)
        digest = pack(staged, archive)
        if args.fixture_out:
            fixture_digest = write_fixture(staged, args.fixture_rows, args.fixture_out)
            print(f"{args.fixture_out}: sha256 {fixture_digest}")
    print(f"{archive}: sha256 {digest}")
    if args.upload:
        print(f"Uploaded to {args.upload} at revision {upload(args.upload, archive)}")


if __name__ == "__main__":
    main()
