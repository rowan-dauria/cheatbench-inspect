"""Build and optionally upload the pinned Creative Writings interviewer data.

    uv run python scripts/creative_writings_dataset.py OUT_DIR --upstream /home/agent/cheatbench
    uv run python scripts/creative_writings_dataset.py OUT_DIR --upload-existing REPO_ID

To build on a single-architecture machine, enable Docker's QEMU emulation first.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from geoguessr_dataset import BASE_IMAGE, UPSTREAM_COMMIT, pack

TASK = "tasks/creative_writings"
ARCHIVE_NAME = "creative_writings.tar.gz"
ARCHITECTURES = {"amd64": "x86_64", "arm64": "aarch64"}

DOCKERFILE = f"""FROM {BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
"""

# Import only the original review constants, without importing its provider
# clients. Make the original compiler choose its Python's actual extension ABI.
BUILD_INTERVIEWER = """
import ast
import runpy
import sys
import sysconfig
from pathlib import Path
from types import ModuleType

source = ast.parse(Path("core/review_server.py").read_text())
review = ModuleType("core.review_server")
for node in source.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in {
            "_MAX_ATTEMPTS", "_CALL_TIMEOUT_S", "CLIENT_TIMEOUT_S", "UNAVAILABLE"
        }:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "review_server.py", "exec"),
                 review.__dict__)
sys.modules["core.review_server"] = review
sys.path.insert(0, "/build/.build-deps")

from tasks.creative_writings import _builder
_builder.SO_NAME = "_score" + sysconfig.get_config_var("EXT_SUFFIX")
sys.argv = ["tasks/creative_writings/build.py", "--variant", "interviewer", *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def build_for_architecture(upstream: Path, dest: Path, platform: str, fixture_row: str) -> Path:
    dest.mkdir()
    archive = subprocess.check_output(
        [
            "git",
            "-C",
            str(upstream),
            "archive",
            UPSTREAM_COMMIT,
            TASK,
            "core/review_server.py",
            "LICENSE",
        ]
    )
    subprocess.run(["tar", "-xf", "-", "-C", str(dest)], input=archive, check=True)
    shutil.copytree(
        Path(yaml.__file__).parent,
        dest / ".build-deps/yaml",
        ignore=shutil.ignore_patterns("*.so", "__pycache__"),
    )
    image = f"writing-builder:{platform}-{hashlib.sha256(DOCKERFILE.encode()).hexdigest()[:12]}"
    subprocess.run(
        ["docker", "build", "--platform", f"linux/{platform}", "-t", image, "-"],
        input=DOCKERFILE,
        text=True,
        check=True,
    )
    flags = [] if platform == "amd64" else ["--row", fixture_row]
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--platform",
            f"linux/{platform}",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{dest}:/build",
            "-w",
            "/build",
            image,
            "python3.11",
            "-c",
            BUILD_INTERVIEWER,
            *flags,
        ],
        check=True,
    )
    return dest / TASK / "environment/interviewer"


def prepare_rows(source: Path, staged: Path, extensions: dict[str, Path], row: str | None) -> None:
    records = [
        json.loads(line)
        for line in (source / "data.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    environment = staged / "environment"
    binaries = environment / "grading/compiled"
    binaries.mkdir(parents=True)
    for extension in extensions.values():
        shutil.copyfile(extension, binaries / extension.name)

    selected = []
    for record in records:
        if row is not None and record["id"] != row:
            continue
        record["files"] = {
            path: relative for path, relative in record["files"].items() if not path.endswith(".so")
        }
        record["score_extensions"] = {
            arch: str((binaries / extension.name).relative_to(environment))
            for arch, extension in extensions.items()
        }
        for relative in (*record["files"].values(), *record["files_abs"].values()):
            target = environment / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        selected.append(record)
    if not selected:
        raise ValueError(f"Requested row {row!r} is missing from the build.")
    (staged / "data.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in selected), encoding="utf-8"
    )


def build(upstream: Path, out_dir: Path, fixture: Path | None) -> None:
    fixture_row = "museum_poem"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor() as executor:
        root = Path(tmp)
        builds = {
            platform: executor.submit(
                build_for_architecture, upstream, root / platform, platform, fixture_row
            )
            for platform in ARCHITECTURES
        }
        sources = {platform: future.result() for platform, future in builds.items()}
        extensions = {
            arch: next(sources[platform].glob("files/grading/**/_score*.so"))
            for platform, arch in ARCHITECTURES.items()
        }
        staged = root / "staged"
        source = sources["amd64"]
        prepare_rows(source, staged, extensions, row=None)
        shutil.copyfile(root / "amd64/LICENSE", staged / "LICENSE")
        archive = out_dir / ARCHIVE_NAME
        print(f"{archive}: SHA-256 {pack(staged, archive, root_files=('data.jsonl', 'LICENSE'))}")
        if fixture is not None:
            fixture.parent.mkdir(parents=True, exist_ok=True)
            selected = root / "fixture"
            prepare_rows(source, selected, extensions, row=fixture_row)
            pack(selected, fixture, root_files=("data.jsonl",))


def upload(repo_id: str, archive: Path) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    card = f"""---
license: mit
pretty_name: CheatBench Creative Writings
---

# CheatBench Creative Writings

All twenty interviewer rows, generated from [CheatBench](https://github.com/centerforaisafety/cheatbench)
at commit `{UPSTREAM_COMMIT}`. The archive contains task-authored briefs and fictional
references, CPython 3.11 grader bytecode, native similarity helpers for Linux amd64
and arm64, and the upstream MIT license.
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
        commit_message=f"Creative Writings interviewer data from CheatBench {UPSTREAM_COMMIT[:7]}",
    )
    print(f"{repo_id}@{commit.oid} SHA-256 {hashlib.sha256(archive.read_bytes()).hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--upstream", type=Path, default=Path("/home/agent/cheatbench"))
    parser.add_argument("--fixture-out", type=Path)
    parser.add_argument("--upload-existing", metavar="REPO_ID")
    args = parser.parse_args()
    if args.upload_existing:
        upload(args.upload_existing, args.out_dir / ARCHIVE_NAME)
    else:
        build(args.upstream, args.out_dir, args.fixture_out)


if __name__ == "__main__":
    main()
