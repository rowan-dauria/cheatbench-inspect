"""Assemble the Prime Factorization task without starting a container."""

import json
from pathlib import Path
from unittest.mock import patch

from inspect_ai import Task

import cheatbench_inspect
from cheatbench_inspect import prime_factorization as prime
from cheatbench_inspect.terminus import TERMINUS_COMPOSE


def test_task_connects_dataset_solver_scorer_and_sandbox(tmp_path: Path) -> None:
    row = {"id": "q4", "input": "Factor 15.", "number": "15", "files": {}, "files_abs": {}}
    (tmp_path / "data.jsonl").write_text(json.dumps(row) + "\n")
    solver = prime.prime_factorization_solver()
    scorer = prime.prime_factorization_scorer()
    with (
        patch.object(prime, "prime_factorization_solver", return_value=solver),
        patch.object(prime, "prime_factorization_scorer", return_value=scorer),
    ):
        task = prime.cheatbench_prime_factorization(task_dir=str(tmp_path))

    assert isinstance(task, Task)
    assert [sample.id for sample in task.dataset] == ["q4"]
    assert task.solver is solver
    assert task.scorer == [scorer]
    assert task.sandbox is not None
    assert task.sandbox.type == "docker"
    assert task.sandbox.config == str(TERMINUS_COMPOSE)


def test_package_exports_prime_factorization_task() -> None:
    assert cheatbench_inspect.cheatbench_prime_factorization is prime.cheatbench_prime_factorization
    assert "cheatbench_prime_factorization" in cheatbench_inspect.__all__


def test_task_defaults_to_first_ten_questions_once(tmp_path: Path) -> None:
    row = {"input": "Factor 15.", "number": "15", "files": {}, "files_abs": {}}
    rows = [{**row, "id": f"fact_n52_{question}"} for question in range(4, 24)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(record) + "\n" for record in rows))

    task = prime.cheatbench_prime_factorization(task_dir=str(tmp_path))
    assert task.epochs == 1
    assert [sample.id for sample in task.dataset] == [record["id"] for record in rows[:10]]

    extended = prime.cheatbench_prime_factorization(task_dir=str(tmp_path), dataset_limit=20)
    assert [sample.id for sample in extended.dataset] == [record["id"] for record in rows]
