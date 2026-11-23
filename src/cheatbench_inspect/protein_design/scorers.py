"""Score Protein Design conversations with the shared judge."""

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from cheatbench_inspect.config import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_REASONING_EFFORT
from cheatbench_inspect.protein_design.resources.judge_schema import (
    EXTRA_CONTEXT,
    ProteinDesignVerdict,
)
from cheatbench_inspect.sandbox_task import trajectory_scorer


@scorer(metrics=[mean()])
def protein_design_scorer() -> Scorer:
    """Judge answer access and final use with the original rubric."""

    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(
            role="grader",
            default=DEFAULT_JUDGE_MODEL,
            config=GenerateConfig(reasoning_effort=DEFAULT_JUDGE_REASONING_EFFORT),
        )
        return await trajectory_scorer(
            ProteinDesignVerdict,
            EXTRA_CONTEXT,
            judge,
            explanation_field="summary",
        )(state, target)

    return score
