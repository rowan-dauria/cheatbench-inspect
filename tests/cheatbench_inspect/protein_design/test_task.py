import hashlib
import json
from functools import partial
from pathlib import Path

import pytest
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver
from tests.cheatbench_inspect.protein_design.conftest import (
    COLLEAGUE_FASTA,
    FIXTURES,
    LOG,
    TARGET_ID,
    WORK_DIR,
    read_fixture,
)

from cheatbench_inspect import cheatbench_protein_design
from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.protein_design import dataset
from cheatbench_inspect.protein_design.protein_design import protein_design_solver
from cheatbench_inspect.protein_design.resources.judge_schema import (
    EXTRA_CONTEXT,
    ProteinDesignVerdict,
)
from cheatbench_inspect.protein_design.scorers import protein_design_scorer
from cheatbench_inspect.sandbox_task import fetch_task_dir


def test_protein_design_task(protein_task_dir: Path) -> None:
    task = cheatbench_protein_design(task_dir=str(protein_task_dir))
    assert task.version == "5-C"
    assert task.message_limit == 300
    assert task.time_limit is None
    assert task.epochs == 1
    [sample] = task.dataset
    assert sample.id == TARGET_ID
    assert sample.input == (FIXTURES / "prompt.txt").read_text()
    assert sample.target == ""
    assert set(sample.files or {}) == {
        f"{WORK_DIR}/target.fasta",
        f"{WORK_DIR}/notes.md",
        LOG,
        COLLEAGUE_FASTA,
    }
    assert sample.metadata
    assert sample.metadata["file_mtimes"][str(Path(COLLEAGUE_FASTA).parent)] == "2026-03-17 16:30"
    assert "/home/user" not in sample.metadata["file_mtimes"]
    assert sample.metadata["pull_log"] == ""
    assert isinstance(protein_design_solver(), Solver)
    assert isinstance(protein_design_scorer(), Scorer)


def test_upstream_context_and_schema() -> None:
    assert read_fixture("extra_context.json") == EXTRA_CONTEXT
    schema = ProteinDesignVerdict.model_json_schema()
    assert schema == read_fixture("shared_verdict_schema.json")
    assert list(schema["properties"]) == schema["required"]
    assert "Use saved-file" in EXTRA_CONTEXT
    assert "grader_state.calls" in schema["properties"]["checks_used"]["description"]


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_public_archive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dataset, "fetch_task_dir", partial(fetch_task_dir, cache_root=tmp_path))
    dataset_samples = cheatbench_protein_design().dataset
    task_dir = dataset.default_task_dir()
    pins = json.loads((task_dir / "targets.json").read_text())
    assert [sample.id for sample in dataset_samples] == [item["id"] for item in pins]
    assert len(dataset_samples) == load_task_defaults("protein_design").dataset_limit == 20
    assert len(dataset.get_dataset(task_dir, limit=1)) == 1
    for sample, pin in zip(dataset_samples, pins, strict=True):
        assert sample.files is not None
        target = next(
            Path(path) for dest, path in sample.files.items() if dest.endswith("/target.fasta")
        )
        header, sequence = target.read_text().splitlines()
        assert header.startswith(f">{pin['entity']}|")
        assert len(sequence) == pin["sequence_length"]
        assert hashlib.sha256(sequence.encode()).hexdigest() == pin["sequence_sha256"]
