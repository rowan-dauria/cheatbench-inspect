"""Shared fixture helpers for Chess tests."""

import tarfile
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def chess_task_dir(tmp_path: Path) -> Path:
    """Unpack the committed one-position LR4pA archive into a temporary directory."""
    target = tmp_path / "chess_data"
    target.mkdir()
    with tarfile.open(FIXTURES / "LR4pA.tar.gz") as archive:
        archive.extractall(target, filter="data")
    return target
