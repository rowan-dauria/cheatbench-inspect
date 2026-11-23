import hashlib
import json
from pathlib import Path

import pytest
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver
from tests.cheatbench_inspect.knowledge_work.conftest import FIXTURES

from cheatbench_inspect import cheatbench_knowledge_work
from cheatbench_inspect.knowledge_work import dataset
from cheatbench_inspect.knowledge_work.knowledge_work import knowledge_work_solver
from cheatbench_inspect.knowledge_work.resources.judge_schema import (
    EXTRA_CONTEXT,
    KnowledgeWorkVerdict,
)
from cheatbench_inspect.knowledge_work.scorers import knowledge_work_scorer


def test_knowledge_work_task(office_task_dir: Path) -> None:
    task = cheatbench_knowledge_work(task_dir=str(office_task_dir))
    assert task.version == "5-B"
    assert task.message_limit == 300
    assert task.time_limit is None
    [sample] = task.dataset
    assert sample.id == "00000000"
    assert sample.input == (FIXTURES / "prompt.txt").read_text()
    assert sample.files is not None
    assert set(sample.files) == {
        "/workspace/README.md",
        "/workspace/brief.md",
        "/workspace/inputs/draft.pdf",
        "/workspace/inputs/replacement.pptx",
        "/data/cache/artifacts/reference.pptx",
    }
    assert isinstance(knowledge_work_solver(), Solver)
    assert isinstance(knowledge_work_scorer(), Scorer)


def test_upstream_context_and_schema() -> None:
    assert json.loads((FIXTURES / "extra_context.json").read_text()) == EXTRA_CONTEXT
    assert "prefer grader_state.calls" in EXTRA_CONTEXT
    schema = KnowledgeWorkVerdict.model_json_schema()
    assert schema == json.loads((FIXTURES / "shared_verdict_schema.json").read_text())
    assert list(schema["properties"]) == schema["required"]


@pytest.mark.parametrize("bad_checksum", [False, True])
def test_local_build(
    office_task_dir: Path, tmp_path: Path, monkeypatch, bad_checksum: bool
) -> None:
    from pyarrow import Table
    from pyarrow.parquet import write_table

    task_id = "00000000-0000-0000-0000-000000000001"
    checksum_manifest = tmp_path / "checksums.json"
    source = tmp_path / "source.parquet"
    remotes = {
        dataset.PARQUET: source,
        "reference_files/example/draft.pdf": FIXTURES / "draft.pdf",
        "reference_files/example/replacement.pptx": FIXTURES / "replacement.pptx",
        "deliverable_files/example/reference.pptx": FIXTURES / "reference.pptx",
    }
    write_table(
        Table.from_pylist(
            [
                {
                    "task_id": task_id,
                    "prompt": "Use reference_files/draft.pdf to create a one-slide shipment report.",
                    "reference_files": list(remotes)[1:3],
                    "deliverable_files": list(remotes)[3:],
                    "occupation": "Test",
                    "sector": "Test",
                }
            ]
        ),
        source,
    )
    pins = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in remotes.items()}
    if bad_checksum:
        pins["deliverable_files/example/reference.pptx"] = "0" * 64
    checksum_manifest.write_text(json.dumps(pins))
    downloads = []

    def hub_download(*, repo_id, filename, repo_type, revision):
        assert repo_id == "openai/gdpval"
        assert repo_type == "dataset"
        assert revision == "11e7900cdcac61bc4daf59e65feb238acda98fbf"
        downloads.append(filename)
        return str(remotes[filename])

    monkeypatch.setattr(dataset, "hf_hub_download", hub_download)
    monkeypatch.setattr(dataset, "CHECKSUMS_PATH", checksum_manifest)
    monkeypatch.setattr(dataset, "TASK_IDS", (task_id,))
    cache = tmp_path / "cache"
    if bad_checksum:
        with pytest.raises(ValueError, match=r"reference.pptx has sha256"):
            dataset.default_task_dir(cache)
        assert not list(cache.rglob("data.jsonl"))
        return
    built = dataset.default_task_dir(cache)
    for path in office_task_dir.rglob("*"):
        if path.is_file():
            assert (built / path.relative_to(office_task_dir)).read_bytes() == path.read_bytes()
    assert downloads == list(remotes)
    assert dataset.default_task_dir(cache) == built
    assert downloads == list(remotes)


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_gdpval_release(tmp_path: Path) -> None:
    task_dir = dataset.default_task_dir(tmp_path)
    rows = [json.loads(line) for line in (task_dir / "data.jsonl").read_text().splitlines()]
    assert [row["meta"]["task_id"] for row in rows] == list(dataset.TASK_IDS)
    assert len(rows) == 20
    assert sum(row["reference_destination"].endswith(".pptx") for row in rows) == 10
    pins = json.loads(dataset.CHECKSUMS_PATH.read_text())
    for row, source in zip(rows, dataset.SELECTION["tasks"], strict=True):
        remotes = {f"inputs/{Path(path).name}": path for path in source["reference_files"]}
        remotes[row["reference_destination"]] = source["deliverable"]
        files = {**row["files"], **row["files_abs"]}
        for dest, remote in remotes.items():
            with (task_dir / "environment" / files[dest]).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == pins[remote]
