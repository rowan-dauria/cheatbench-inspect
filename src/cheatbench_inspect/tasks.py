from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_evals_utils.metadata import load_eval_version

from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.geoguessr import dataset as geo_dataset
from cheatbench_inspect.geoguessr import geoguessr
from cheatbench_inspect.geoguessr.scorers import geoguessr_scorer
from cheatbench_inspect.harnesses.harness import validate_harness
from cheatbench_inspect.harnesses.terminus import TERMINUS_COMPOSE
from cheatbench_inspect.knowledge_work import dataset as work_dataset
from cheatbench_inspect.knowledge_work import knowledge_work
from cheatbench_inspect.knowledge_work.scorers import knowledge_work_scorer
from cheatbench_inspect.prime_factorization import dataset as prime_dataset
from cheatbench_inspect.prime_factorization import prime_factorization as prime
from cheatbench_inspect.prime_factorization import scorers as prime_scorers
from cheatbench_inspect.subset_sum import dataset as subset_dataset
from cheatbench_inspect.subset_sum import subset_sum
from cheatbench_inspect.subset_sum.scorers import subset_sum_scorer
from cheatbench_inspect.sycophancy import sycophancy
from cheatbench_inspect.sycophancy.scorers import sycophancy_scorer
from cheatbench_inspect.synthetic_puzzles import dataset as puzzle_dataset
from cheatbench_inspect.synthetic_puzzles import synthetic_puzzles
from cheatbench_inspect.synthetic_puzzles.scorers import synthetic_puzzles_scorer


def _max_turns(task_name: str) -> int:
    value = load_task_defaults(task_name).max_turns
    if value is None:
        raise ValueError(f"Missing max_turns default for {task_name}")
    return value


@task
def cheatbench_sycophancy(
    prompt_template: Literal["instruction", "dontcheat", "diy"] = "instruction",
) -> Task:
    if prompt_template not in sycophancy.PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown prompt_template {prompt_template!r}; "
            f"choose from {sorted(sycophancy.PROMPT_TEMPLATES)}"
        )
    return Task(
        dataset=sycophancy.get_dataset(prompt_template),
        solver=generate(),
        scorer=sycophancy_scorer(),
        epochs=load_task_defaults("sycophancy").epochs,
        version=load_eval_version(__file__),
    )


_HARNESS_COMPOSE = {
    "terminus-2": TERMINUS_COMPOSE,
    "claude-code": TERMINUS_COMPOSE.with_name("claude-compose.yaml"),
    "codex": TERMINUS_COMPOSE.with_name("codex-compose.yaml"),
}


_OFFLINE_HARNESS_COMPOSE = {
    "terminus-2": TERMINUS_COMPOSE.with_name("terminus-offline-compose.yaml"),
    "claude-code": TERMINUS_COMPOSE.with_name("claude-offline-compose.yaml"),
    "codex": TERMINUS_COMPOSE.with_name("codex-offline-compose.yaml"),
}


_OFFICE_HARNESS_COMPOSE = {
    "terminus-2": TERMINUS_COMPOSE.with_name("terminus-office-compose.yaml"),
    "claude-code": TERMINUS_COMPOSE.with_name("claude-office-compose.yaml"),
    "codex": TERMINUS_COMPOSE.with_name("codex-office-compose.yaml"),
}


@task
def cheatbench_prime_factorization(
    task_dir: str | None = None,
    dataset_limit: int | None = None,
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Task:
    """Run Prime Factorization with the selected agent harness.

    Args:
        task_dir: Directory containing the generated data.jsonl and environment/ tree.
            Defaults to the published build of the original's builder, downloaded and cached.
        dataset_limit: Number of questions to load from the start of data.jsonl.
        harness: Agent harness to run: terminus-2, claude-code, or codex.
        harness_version: Claude CLI or Codex version, required for claude-code and codex.
        harness_model: Original model ID for the Claude Code or Codex CLI and judge transcript.
    """
    validate_harness(harness)
    if harness in {"claude-code", "codex"} and (not harness_version or not harness_model):
        raise ValueError(f"{harness} requires harness_version and harness_model")
    if harness == "terminus-2" and (harness_version is not None or harness_model is not None):
        raise ValueError("terminus-2 does not accept harness_version or harness_model")
    if dataset_limit is None:
        dataset_limit = load_task_defaults("prime_factorization").dataset_limit
    assert dataset_limit is not None
    if dataset_limit < 1:
        raise ValueError("dataset_limit must be positive.")
    data_dir = Path(task_dir) if task_dir is not None else prime_dataset.default_task_dir()
    return Task(
        dataset=prime_dataset.get_dataset(data_dir, limit=dataset_limit),
        scorer=prime_scorers.prime_factorization_scorer(),
        solver=prime.prime_factorization_solver(harness, harness_version, harness_model),
        sandbox=("docker", str(_HARNESS_COMPOSE[harness])),
        # One assistant message and one tool message per turn.
        message_limit=2 * _max_turns("prime_factorization"),
        epochs=load_task_defaults("prime_factorization").epochs,
        version=load_eval_version(__file__),
    )


@task
def cheatbench_subset_sum(
    task_dir: str | None = None,
    dataset_limit: int | None = None,
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Task:
    """Run Subset Sum with the selected agent harness.

    Args:
        task_dir: Directory containing the generated data.jsonl and environment/ tree.
            Defaults to the pinned Hugging Face archive, downloaded and cached.
        dataset_limit: Number of questions to load from the start of data.jsonl.
        harness: Agent harness to run: terminus-2, claude-code, or codex.
        harness_version: Claude CLI or Codex version, required for claude-code and codex.
        harness_model: Claude CLI or Codex model ID, required for claude-code and codex.
    """
    validate_harness(harness)
    if harness in {"claude-code", "codex"} and (not harness_version or not harness_model):
        raise ValueError(f"{harness} requires harness_version and harness_model")
    if harness == "terminus-2" and (harness_version is not None or harness_model is not None):
        raise ValueError("terminus-2 does not accept harness_version or harness_model")
    if dataset_limit is None:
        dataset_limit = load_task_defaults("subset_sum").dataset_limit
    assert dataset_limit is not None
    if dataset_limit < 1:
        raise ValueError("dataset_limit must be positive.")
    data_dir = Path(task_dir) if task_dir is not None else subset_dataset.default_task_dir()
    return Task(
        dataset=subset_dataset.get_dataset(data_dir, limit=dataset_limit),
        scorer=subset_sum_scorer(),
        solver=subset_sum.subset_sum_solver(harness, harness_version, harness_model),
        sandbox=("docker", str(_HARNESS_COMPOSE[harness])),
        # One assistant message and one tool message per turn.
        message_limit=2 * _max_turns("subset_sum"),
        epochs=load_task_defaults("subset_sum").epochs,
        version=load_eval_version(__file__),
    )


@task
def cheatbench_geoguessr(
    task_dir: str | None = None,
    dataset_limit: int | None = None,
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Task:
    """Run GeoGuessr with the selected agent harness.

    Args:
        task_dir: Directory containing the generated data.jsonl, manifest.json and environment/ tree.
            Defaults to the pinned Hugging Face archive, downloaded and cached.
        dataset_limit: Number of photographs to load from the start of data.jsonl.
        harness: Agent harness to run.
        harness_version: Claude CLI or Codex version, required for claude-code and codex.
        harness_model: Claude CLI or Codex model ID, required for claude-code and codex.
    """
    validate_harness(harness)
    if harness in {"claude-code", "codex"} and (not harness_version or not harness_model):
        raise ValueError(f"{harness} requires harness_version and harness_model")
    if harness == "terminus-2" and (harness_version is not None or harness_model is not None):
        raise ValueError("terminus-2 does not accept harness_version or harness_model")
    if dataset_limit is None:
        dataset_limit = load_task_defaults("geoguessr").dataset_limit
    assert dataset_limit is not None
    if dataset_limit < 1:
        raise ValueError("dataset_limit must be positive.")
    data_dir = Path(task_dir) if task_dir is not None else geo_dataset.default_task_dir()
    return Task(
        dataset=geo_dataset.get_dataset(data_dir, limit=dataset_limit),
        scorer=geoguessr_scorer(data_dir),
        solver=geoguessr.geoguessr_solver(harness, harness_version, harness_model),
        sandbox=("docker", str(_OFFLINE_HARNESS_COMPOSE[harness])),
        # One assistant message and one tool message per turn.
        message_limit=2 * _max_turns("geoguessr"),
        epochs=load_task_defaults("geoguessr").epochs,
        version=load_eval_version(__file__),
    )


@task
def cheatbench_synthetic_puzzles(
    task_dir: str | None = None,
    dataset_limit: int | None = None,
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Task:
    """Run Synthetic Puzzles with the selected agent harness.

    Args:
        task_dir: Directory containing the generated data.jsonl and environment/ tree.
            Defaults to the pinned Hugging Face archive, downloaded and cached.
        dataset_limit: Number of puzzles to load from the start of data.jsonl.
        harness: Agent harness to run.
        harness_version: Claude CLI or Codex version, required for claude-code and codex.
        harness_model: Claude CLI or Codex model ID, required for claude-code and codex.
    """
    validate_harness(harness)
    if harness in {"claude-code", "codex"} and (not harness_version or not harness_model):
        raise ValueError(f"{harness} requires harness_version and harness_model")
    if harness == "terminus-2" and (harness_version is not None or harness_model is not None):
        raise ValueError("terminus-2 does not accept harness_version or harness_model")
    if dataset_limit is None:
        dataset_limit = load_task_defaults("synthetic_puzzles").dataset_limit
    assert dataset_limit is not None
    if dataset_limit < 1:
        raise ValueError("dataset_limit must be positive.")
    data_dir = Path(task_dir) if task_dir is not None else puzzle_dataset.default_task_dir()
    return Task(
        dataset=puzzle_dataset.get_dataset(data_dir, limit=dataset_limit),
        scorer=synthetic_puzzles_scorer(data_dir),
        solver=synthetic_puzzles.synthetic_puzzles_solver(harness, harness_version, harness_model),
        sandbox=("docker", str(_OFFLINE_HARNESS_COMPOSE[harness])),
        # One assistant message and one tool message per turn.
        message_limit=2 * _max_turns("synthetic_puzzles"),
        epochs=load_task_defaults("synthetic_puzzles").epochs,
        version=load_eval_version(__file__),
    )


@task
def cheatbench_knowledge_work(
    task_dir: str | None = None,
    dataset_limit: int | None = None,
    harness: str = "terminus-2",
    harness_version: str | None = None,
    harness_model: str | None = None,
) -> Task:
    """Run Knowledge Work with the selected agent harness.

    Args:
        task_dir: Directory containing the generated data.jsonl and environment/ tree.
            Defaults to a local build from the pinned GDPval revision.
        dataset_limit: Number of assignments to load from the start of data.jsonl.
        harness: Agent harness to run.
        harness_version: Claude CLI or Codex version, required for claude-code and codex.
        harness_model: Claude CLI or Codex model ID, required for claude-code and codex.
    """
    validate_harness(harness)
    if harness in {"claude-code", "codex"} and (not harness_version or not harness_model):
        raise ValueError(f"{harness} requires harness_version and harness_model")
    if harness == "terminus-2" and (harness_version is not None or harness_model is not None):
        raise ValueError("terminus-2 does not accept harness_version or harness_model")
    if dataset_limit is None:
        dataset_limit = load_task_defaults("knowledge_work").dataset_limit
    assert dataset_limit is not None
    if dataset_limit < 1:
        raise ValueError("dataset_limit must be positive.")
    data_dir = Path(task_dir) if task_dir is not None else work_dataset.default_task_dir()
    return Task(
        dataset=work_dataset.get_dataset(data_dir, limit=dataset_limit),
        scorer=knowledge_work_scorer(),
        solver=knowledge_work.knowledge_work_solver(harness, harness_version, harness_model),
        sandbox=("docker", str(_OFFICE_HARNESS_COMPOSE[harness])),
        # One assistant message and one tool message per turn.
        message_limit=2 * _max_turns("knowledge_work"),
        epochs=load_task_defaults("knowledge_work").epochs,
        version=load_eval_version(__file__),
    )
