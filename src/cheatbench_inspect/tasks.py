from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_evals_utils.metadata import load_eval_version

from cheatbench_inspect.harnesses.harness import validate_harness
from cheatbench_inspect.harnesses.terminus import TERMINUS_COMPOSE
from cheatbench_inspect.prime_factorization import dataset as prime_dataset
from cheatbench_inspect.prime_factorization import prime_factorization as prime
from cheatbench_inspect.prime_factorization import scorers as prime_scorers
from cheatbench_inspect.sycophancy import sycophancy
from cheatbench_inspect.sycophancy.scorers import sycophancy_scorer


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
        epochs=sycophancy.DEFAULT_EPOCHS,
        version=load_eval_version(__file__),
    )


_HARNESS_COMPOSE = {
    "terminus-2": TERMINUS_COMPOSE,
    "claude-code": TERMINUS_COMPOSE.with_name("claude-compose.yaml"),
    "codex": TERMINUS_COMPOSE.with_name("codex-compose.yaml"),
}


@task
def cheatbench_prime_factorization(
    task_dir: str | None = None,
    dataset_limit: int = prime_dataset.DEFAULT_DATASET_LIMIT,
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
    if dataset_limit < 1:
        raise ValueError("dataset_limit must be positive.")
    data_dir = Path(task_dir) if task_dir is not None else prime_dataset.default_task_dir()
    return Task(
        dataset=prime_dataset.get_dataset(data_dir, limit=dataset_limit),
        scorer=prime_scorers.prime_factorization_scorer(),
        solver=prime.prime_factorization_solver(harness, harness_version, harness_model),
        sandbox=("docker", str(_HARNESS_COMPOSE[harness])),
        # One assistant message and one tool message per turn.
        message_limit=2 * prime.MAX_TURNS,
        epochs=prime.DEFAULT_EPOCHS,
        version=load_eval_version(__file__),
    )
