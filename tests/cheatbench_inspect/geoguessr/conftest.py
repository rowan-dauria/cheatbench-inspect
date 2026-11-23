from pathlib import Path

import pytest

from cheatbench_inspect.sandbox_task import unpack_archive


@pytest.fixture
def row0_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "row0"
    unpack_archive(Path(__file__).parent / "fixtures/geoguessr_31fee33fbdbd6b87.tar.gz", task_dir)
    return task_dir
