from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from cheatbench_inspect.config import load_task_defaults


def test_defaults_and_metadata():
    path = Path(__file__).parents[2] / "src/cheatbench_inspect/eval.yaml"
    data = yaml.safe_load(path.read_text())
    assert set(data["metadata"]) == {"extras"}
    for task, limit, timeout in [
        ("prime_factorization", 10, 3600),
        ("subset_sum", 10, 3600),
        ("geoguessr", 20, 3600),
        ("synthetic_puzzles", 20, 1800),
    ]:
        defaults = load_task_defaults(task)
        assert defaults.epochs == 1
        assert defaults.max_turns == 150
        assert defaults.timeout_s == timeout
        assert defaults.dataset_limit == limit
    assert load_task_defaults("sycophancy").epochs == 100


def test_invalid_defaults():
    with pytest.raises(ValueError, match="Unknown task"):
        load_task_defaults("missing")
    with (
        patch(
            "cheatbench_inspect.config._task_defaults_data",
            return_value={"task": {"epochs": 1, "bad": 2}},
        ),
        pytest.raises(ValueError, match="Unknown defaults"),
    ):
        load_task_defaults("task")
    with (
        patch(
            "cheatbench_inspect.config._task_defaults_data", return_value={"task": {"epochs": True}}
        ),
        pytest.raises(ValueError, match="integers"),
    ):
        load_task_defaults("task")
