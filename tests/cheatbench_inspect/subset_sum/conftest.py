from pathlib import Path

import pytest

from cheatbench_inspect.sandbox_task import unpack_archive


@pytest.fixture
def q4_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "q4"
    unpack_archive(Path(__file__).parent / "fixtures/q4.tar.gz", task_dir)
    return task_dir
