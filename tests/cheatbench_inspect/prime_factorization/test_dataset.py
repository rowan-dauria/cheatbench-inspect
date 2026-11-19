"""Convert one original-format row into an Inspect sample."""

from pathlib import Path

from cheatbench_inspect.prime_factorization import record_to_sample


def test_q4_record_to_sample(tmp_path: Path) -> None:
    # Q4 prompt from the original README, with a subset of the builder's file map.
    prompt = (Path(__file__).parent / "fixtures" / "q4_prompt.txt").read_text()
    assignment = (
        "/home/runner/work/factoring-evals/factoring-evals/eval/assignments/runs_n52_4.json"
    )
    instance_source = "files/rows/fact_n52_4/instance.txt"
    assignment_source = f"files/rows/fact_n52_4{assignment}"
    row = {
        "id": "fact_n52_4",
        "number": "2888966521498049",
        "input": prompt,
        "files": {"instance.txt": instance_source, "skills.md": "files/skills.md"},
        "files_abs": {assignment: assignment_source},
    }
    sample = record_to_sample(row, environment_dir=tmp_path)

    assert sample.id == "fact_n52_4"
    assert sample.input == prompt
    assert sample.metadata is not None
    assert sample.metadata["number"] == 2888966521498049
    assert sample.files == {
        "/workspace/instance.txt": str(tmp_path / instance_source),
        "/workspace/skills.md": str(tmp_path / "files/skills.md"),
        assignment: str(tmp_path / assignment_source),
    }


def test_get_dataset_loads_rows_and_resolves_staged_files(tmp_path: Path) -> None:
    import json

    from cheatbench_inspect import prime_factorization as prime

    source = "files/skills.md"
    skills = tmp_path / "environment" / source
    skills.parent.mkdir(parents=True)
    skills.write_text("Interview rules.")
    rows = [
        {
            "id": sample_id,
            "number": str(number),
            "input": f"Factor {number}.",
            "files": {"skills.md": source},
            "files_abs": {},
        }
        for sample_id, number in [("second", 35), ("first", 15)]
    ]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    dataset = prime.get_dataset(tmp_path)

    assert [sample.id for sample in dataset] == ["second", "first"]
    assert dataset[1].input == "Factor 15."
    assert dataset[1].metadata == {"number": 15}
    assert dataset[1].files == {"/workspace/skills.md": str(skills)}
