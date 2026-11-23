import json
import shutil
from pathlib import Path

import pytest

from cheatbench_inspect.knowledge_work.dataset import RESOURCES

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def office_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    row = json.loads((FIXTURES / "row.json").read_text())
    (task_dir / "data.jsonl").write_text(json.dumps(row) + "\n")
    for destination, relative in {**row["files"], **row["files_abs"]}.items():
        target = task_dir / "environment" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if destination == "README.md":
            shutil.copyfile(RESOURCES / "workspace_readme.txt", target)
        elif destination == "brief.md":
            target.write_text("Use inputs/draft.pdf to create a one-slide shipment report.\n")
        else:
            shutil.copyfile(FIXTURES / Path(relative).name, target)
    return task_dir
