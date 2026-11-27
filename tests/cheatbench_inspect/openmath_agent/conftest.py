from pathlib import Path

import pytest

from cheatbench_inspect.sandbox_task import unpack_archive

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    task = tmp_path / "data"
    unpack_archive(FIXTURES / "task.tar.gz", task)
    return task
