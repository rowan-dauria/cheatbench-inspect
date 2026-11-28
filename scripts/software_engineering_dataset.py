"""Build or verify a local Software Engineering (SWE-bench Pro) task directory.

The task's 20 SWE-bench Pro instances contain problem statements, patches and
test scripts derived from 11 GPL-3.0 / AGPL-3.0 / upstream-licensed repositories,
so this repository ships only the pinned pointers, Docker Hub digests and
SHA-256 manifest in `src/cheatbench_inspect/software_engineering/resources/instances.json`.

    uv run python scripts/software_engineering_dataset.py OUT_DIR
    uv run python scripts/software_engineering_dataset.py OUT_DIR --upstream-tasks /path/to/harbor/tasks
    uv run python scripts/software_engineering_dataset.py --check --upstream-tasks /path/to/harbor/tasks
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

from cheatbench_inspect.software_engineering.dataset import (
    TEST_FILENAMES,
    build_task_dir,
    download_instance_solution,
    load_manifest,
)


def check_manifest(*, upstream_tasks_dir: Path | None) -> None:
    """Verify every pinned file and reference solution SHA-256 in instances.json."""
    manifest = load_manifest()
    with tempfile.TemporaryDirectory(prefix="swe-check-") as temporary:
        out = Path(temporary)
        build_task_dir(out, upstream_tasks_dir=upstream_tasks_dir)
        for instance in manifest["instances"]:
            instance_id = instance["id"]
            download_instance_solution(instance_id, upstream_tasks_dir=upstream_tasks_dir)
            for filename in TEST_FILENAMES:
                path = out / "instances" / instance_id / "tests" / filename
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                expected = instance["sha256"][f"tests/{filename}"]
                if digest != expected:
                    raise SystemExit(f"{instance_id}/tests/{filename}: {digest} != {expected}")
    print(f"Verified all {len(manifest['instances'])} instances against instances.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        help="Directory to write data.jsonl and per-instance prompts and verifier files.",
    )
    parser.add_argument(
        "--upstream-tasks",
        type=Path,
        help="Read the 20 instances from a local CheatBench harbor/tasks directory instead of GitHub.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify every instance file and solution/solve.sh against instances.json.",
    )
    args = parser.parse_args()
    if args.check:
        check_manifest(upstream_tasks_dir=args.upstream_tasks)
        return
    if args.out_dir is None:
        parser.error("out_dir is required unless --check is set")
    build_task_dir(args.out_dir, upstream_tasks_dir=args.upstream_tasks)
    print(f"Built Software Engineering task directory at {args.out_dir}")


if __name__ == "__main__":
    main()
