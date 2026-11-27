import json
from pathlib import Path

import pytest

from cheatbench_inspect.sandbox_task import unpack_archive

FIXTURES = Path(__file__).with_name("fixtures")
LEADERBOARD = json.loads((FIXTURES / "leaderboard.json").read_text(encoding="utf-8"))
VERDICTS = json.loads((FIXTURES / "verdicts.json").read_text(encoding="utf-8"))
RUBRIC = "/codabench/competitions/517b32e2/rubric.json"


@pytest.fixture
def svg_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    unpack_archive(FIXTURES / "workshop_fixture.tar.gz", task_dir)
    return task_dir
