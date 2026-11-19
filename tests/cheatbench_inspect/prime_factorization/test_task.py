"""Assemble the Prime Factorization task without starting a container."""

import json
from pathlib import Path
from unittest.mock import patch

from inspect_ai import Task

from cheatbench_inspect import prime_factorization as prime
from cheatbench_inspect.terminus import TERMINUS_COMPOSE


def test_task_uses_prime_dataset_solver_and_terminus_sandbox(tmp_path: Path) -> None:
    row = {"id": "q4", "input": "Factor 15.", "number": "15", "files": {}, "files_abs": {}}
    (tmp_path / "data.jsonl").write_text(json.dumps(row) + "\n")
    solver = prime.prime_factorization_solver()
    with patch.object(prime, "prime_factorization_solver", return_value=solver):
        task = prime.cheatbench_prime_factorization(task_dir=str(tmp_path))

    assert isinstance(task, Task)
    assert [sample.id for sample in task.dataset] == ["q4"]
    assert task.solver is solver
    assert task.sandbox is not None
    assert task.sandbox.type == "docker"
    assert task.sandbox.config == str(TERMINUS_COMPOSE)
