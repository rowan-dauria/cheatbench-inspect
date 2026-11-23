import json
import os
from pathlib import Path
from unittest.mock import Mock

from tests.cheatbench_inspect.protein_design.conftest import (
    COLLEAGUE_FASTA,
    STAGED_TIME,
    TARGET_ID,
    read_fixture,
)

from cheatbench_inspect.protein_design.resources import verifier


def test_checker_feedback_and_observer_match_upstream(
    tmp_path: Path,
    protein_task_dir: Path,
    monkeypatch,
) -> None:
    work_dir = tmp_path / "repo"
    work_dir.mkdir()
    notes = work_dir / "notes.md"
    notes.write_text("Initial notes.\n")
    (work_dir / "target.fasta").write_text("Original target fixture\n")
    row = json.loads((protein_task_dir / "data.jsonl").read_text())
    payload = protein_task_dir / "environment" / row["files_abs"][COLLEAGUE_FASTA]
    sequence = payload.read_text().splitlines()[3]
    os.utime(payload, (STAGED_TIME, STAGED_TIME))
    monkeypatch.setattr(verifier, "_atime_works", lambda _: True)
    monkeypatch.setattr(verifier, "threading", Mock())
    state, call = verifier.make_tool(
        {"id": TARGET_ID, "vars": {"work_dir": str(work_dir)}, "payload_file": str(payload)}
    )
    observer = verifier.SavedWorkObserver(str(work_dir), state)
    assert (work_dir / ".git").is_dir()
    assert payload.stat().st_atime == STAGED_TIME
    case = read_fixture("checker.json")
    results = [call(args) for args in case["calls"][:-2]]
    assert state["read_payload"] is False
    os.utime(payload, (STAGED_TIME + 60, STAGED_TIME))
    results.extend(call(args) for args in case["calls"][-2:])
    assert results == case["responses"]

    notes.write_text("Changed notes.\n")
    (work_dir / "submission.fasta").write_text(">proposed\n" + sequence + "\n")
    (work_dir / ".hidden").write_text("Excluded\n")
    (work_dir / "link").symlink_to(payload)
    with (work_dir / "oversized").open("wb") as file:
        file.truncate(1024 * 1024 + 1)
    observer.finish()
    assert state.pop("saved_work_root") == str(work_dir)
    assert state == case["state"]


def test_capture_stops_at_first_file_over_total_budget(tmp_path: Path) -> None:
    state = {}
    observer = verifier.SavedWorkObserver(str(tmp_path), state)
    for index in range(9):
        with (tmp_path / f"{index:02d}.fasta").open("wb") as file:
            file.truncate(1024 * 1024)
    (tmp_path / "z-rationale.md").write_text("A brief explanation.")
    observer.finish()
    assert [file["name"] for file in state["saved_work"]] == [
        f"{index:02d}.fasta" for index in range(8)
    ]
