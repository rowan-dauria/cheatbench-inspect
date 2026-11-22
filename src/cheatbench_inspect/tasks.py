from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_evals_utils.metadata import load_eval_version

from cheatbench_inspect import prime_factorization as prime
from cheatbench_inspect import sycophancy
from cheatbench_inspect.harness import validate_harness
from cheatbench_inspect.terminus import TERMINUS_COMPOSE


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
        scorer=sycophancy.sycophancy_scorer(),
        epochs=sycophancy.DEFAULT_EPOCHS,
        version=load_eval_version(__file__),
    )


@task
def cheatbench_prime_factorization(
    task_dir: str | None = None,
    dataset_limit: int = prime.DEFAULT_DATASET_LIMIT,
    harness: str = "terminus-2",
) -> Task:
    """Run Prime Factorization with the Terminus-2 harness.

    Args:
        task_dir: Directory containing the generated data.jsonl and environment/ tree.
            Defaults to the published build of the original's builder, downloaded and cached.
        dataset_limit: Number of questions to load from the start of data.jsonl.
        harness: Agent harness to run.
    """
    validate_harness(harness)
    if dataset_limit < 1:
        raise ValueError("dataset_limit must be positive.")
    data_dir = Path(task_dir) if task_dir is not None else prime.default_task_dir()
    return Task(
        dataset=prime.get_dataset(data_dir, limit=dataset_limit),
        scorer=prime.prime_factorization_scorer(),
        solver=prime.prime_factorization_solver(harness),
        sandbox=("docker", str(TERMINUS_COMPOSE)),
        epochs=prime.DEFAULT_EPOCHS,
        version=load_eval_version(__file__),
    )
