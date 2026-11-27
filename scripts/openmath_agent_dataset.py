"""Build a deterministic local OpenMath Agent archive from the pinned original."""

import argparse
import hashlib
import tempfile
from pathlib import Path

from cheatbench_inspect.openmath.build import write_archive
from cheatbench_inspect.openmath_agent.build import build_task_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--upstream-task", type=Path, help="Read pinned builder files locally.")
    parser.add_argument("--assets", type=Path, help="Reuse the builder's verified download cache.")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="research-") as temporary:
        task_dir = Path(temporary) / "data"
        build_task_dir(task_dir, upstream_task=args.upstream_task, assets=args.assets)
        write_archive(args.archive, task_dir, sorted(task_dir.iterdir()), preserve_modes=True)
    with args.archive.open("rb") as stream:
        print(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {args.archive}")


if __name__ == "__main__":
    main()
