"""Build the pinned Knowledge Work dataset without publishing its documents."""

import argparse
from pathlib import Path

from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH

from cheatbench_inspect.knowledge_work.dataset import default_task_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=INSPECT_EVALS_CACHE_PATH)
    args = parser.parse_args()
    print(default_task_dir(args.cache_root))


if __name__ == "__main__":
    main()
