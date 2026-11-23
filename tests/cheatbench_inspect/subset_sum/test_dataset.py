import json
from pathlib import Path

from cheatbench_inspect.harnesses.container import agent_owned_roots
from cheatbench_inspect.subset_sum import dataset as subset_sum


def test_q4_dataset_and_task_defaults(tmp_path: Path) -> None:
    row = json.loads((Path(__file__).parent / "fixtures/q4.json").read_text())
    rows = [{**row, "id": f"subsetsum_n60d10_{i}"} for i in range(4, 16)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    data = subset_sum.get_dataset(tmp_path, limit=10)
    assert [s.id for s in data] == [r["id"] for r in rows[:10]]
    sample = data[0]
    assert sample.metadata is not None
    assert sample.files is not None
    assert sample.input == row["input"]
    assert sample.metadata["numbers"] == row["numbers"]
    assert sample.metadata["target"] == 163721718194
    assert "live_logs" not in sample.metadata
    assert len(sample.files) == 283
    assert sample.metadata["staged_files"] == sorted(sample.files)
    assert agent_owned_roots(sample.metadata["staged_files"]) == ["/home/runner", "/workspace"]
    assert sample.files["/workspace/instance.txt"] == str(
        tmp_path / "environment" / row["files"]["instance.txt"]
    )


def test_q4_workspace_files_are_builder_outputs(q4_task_dir: Path) -> None:
    fixture = Path(__file__).parent / "fixtures"
    row = json.loads((fixture / "q4.json").read_text())
    sample = subset_sum.record_to_sample(row, environment_dir=q4_task_dir / "environment")
    assert sample.files is not None
    instance = Path(sample.files["/workspace/instance.txt"]).read_text(encoding="utf-8")
    assert "TYPE: SUBSET_SUM" in instance
    assert "TARGET: 163721718194" in instance
    assert Path(sample.files["/workspace/skills.md"]).is_file()


def test_q4_fixture_contains_all_staged_files(q4_task_dir: Path):
    fixture = Path(__file__).parent / "fixtures"
    row = json.loads((fixture / "q4.json").read_text())
    generated = json.loads((q4_task_dir / "data.jsonl").read_text())
    assert generated == row
    paths = set(row["files"].values()) | set(row["files_abs"].values())
    assert all((q4_task_dir / "environment" / path).is_file() for path in paths)
