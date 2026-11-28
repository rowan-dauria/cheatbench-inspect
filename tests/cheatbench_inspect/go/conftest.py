"""Shared fixture helpers for Go tests."""

import tarfile
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def go_task_dir(tmp_path: Path) -> Path:
    """Unpack the committed one-position ogs_69046172 archive into a temporary directory."""
    target = tmp_path / "go_data"
    target.mkdir()
    with tarfile.open(FIXTURES / "ogs_69046172.tar.gz") as archive:
        archive.extractall(target, filter="data")
    return target
