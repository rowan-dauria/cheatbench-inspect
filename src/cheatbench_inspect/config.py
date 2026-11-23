"""Default settings shared by CheatBench tasks."""

from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path

import yaml

# Original CheatBench core/judge.py and the gpt-6-sol profile in configs/models.yaml.
DEFAULT_JUDGE_MODEL = "openai/gpt-6-sol"
DEFAULT_JUDGE_REASONING_EFFORT = "high"


@dataclass(frozen=True)
class TaskDefaults:
    epochs: int
    max_turns: int | None = None
    timeout_s: int | None = None
    dataset_limit: int | None = None


@cache
def _task_defaults_data() -> dict:
    with (Path(__file__).parent / "eval.yaml").open() as file:
        return yaml.safe_load(file)["metadata"]["extras"]["task_defaults"]


def load_task_defaults(task: str) -> TaskDefaults:
    data = _task_defaults_data()
    if task not in data:
        raise ValueError(f"Unknown task defaults: {task}")
    values = data[task]
    if not isinstance(values, dict):
        raise ValueError(f"Invalid defaults for {task}: expected a mapping")
    unknown = values.keys() - {field.name for field in fields(TaskDefaults)}
    if unknown:
        raise ValueError(f"Unknown defaults for {task}: {sorted(unknown)}")
    if "epochs" not in values or any(type(value) is not int for value in values.values()):
        raise ValueError(
            f"Invalid defaults for {task}: values must be integers and epochs is required"
        )
    return TaskDefaults(**values)
