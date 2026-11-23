from pathlib import Path

import pytest

from cheatbench_inspect.sandbox_task import unpack_archive

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def puzzle_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "puzzle-fixture"
    unpack_archive(FIXTURES / "puzzle_img_01.tar.gz", task_dir)
    # The checker uses the image's basename. Wire tests use the real JPEG from the archive.
    image = task_dir / "environment/files/workspace/puzzle/img_01.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.touch()
    return task_dir
