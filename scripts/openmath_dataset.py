"""Build a local, deterministic OpenMath task archive from the pinned original."""

import argparse
import hashlib
import tempfile
from pathlib import Path

from cheatbench_inspect.openmath.build import build_task_dir, write_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--upstream-task", type=Path, help="Read the pinned inputs from a checkout."
    )
    parser.add_argument(
        "--regenerate-materials",
        type=Path,
        help="Run the original Info-ZIP build and write a replacement materials bundle.",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="openmath-") as temporary:
        task_dir = Path(temporary)
        build_task_dir(
            task_dir,
            upstream_task=args.upstream_task,
            regenerate_materials=args.regenerate_materials,
        )
        write_archive(args.archive, task_dir, sorted(task_dir.iterdir()))
    with args.archive.open("rb") as stream:
        print(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {args.archive}")


if __name__ == "__main__":
    main()
