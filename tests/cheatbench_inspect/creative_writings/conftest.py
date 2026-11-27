from pathlib import Path

import pytest

from cheatbench_inspect.sandbox_task import unpack_archive

FIXTURES = Path(__file__).with_name("fixtures")


@pytest.fixture
def writing_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "writing"
    unpack_archive(FIXTURES / "museum_poem.tar.gz", task_dir)
    return task_dir
