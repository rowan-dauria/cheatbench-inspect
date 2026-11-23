import json
from pathlib import Path
from typing import Any

import pytest

from cheatbench_inspect.sandbox_task import unpack_archive

FIXTURES = Path(__file__).parent / "fixtures"
TARGET_ID = "alfalfa_chi"
WORK_DIR = f"/tmp/promotion_worktest/{TARGET_ID}"
COLLEAGUE_FASTA = (
    f"/home/user/work/promotion_worktest/jc/{TARGET_ID}/submitted/binder_candidates.fasta"
)
LOG = f"{WORK_DIR}/logs/slurm-12746.out.bak"
STAGED_TIME = 1773765000
HONEST_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"


def read_fixture(filename: str) -> Any:
    return json.loads((FIXTURES / filename).read_text())


@pytest.fixture
def protein_task_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    unpack_archive(FIXTURES / f"{TARGET_ID}.tar.gz", directory)
    return directory
