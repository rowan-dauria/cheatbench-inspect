from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_evals_utils.metadata import load_eval_version

from cheatbench_inspect import prime_factorization as prime
from cheatbench_inspect import sycophancy
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
    task_dir: str, dataset_limit: int = prime.DEFAULT_DATASET_LIMIT
) -> Task:
    """Run Prime Factorization with the Terminus-2 harness.

    Args:
        task_dir: Directory containing the generated data.jsonl and environment/ tree.
        dataset_limit: Number of questions to load from the start of data.jsonl.
    """
    if dataset_limit < 1:
        raise ValueError("dataset_limit must be positive.")
    return Task(
        dataset=prime.get_dataset(Path(task_dir), limit=dataset_limit),
        scorer=prime.prime_factorization_scorer(),
        solver=prime.prime_factorization_solver(),
        sandbox=("docker", str(TERMINUS_COMPOSE)),
        epochs=prime.DEFAULT_EPOCHS,
        version=load_eval_version(__file__),
    )
