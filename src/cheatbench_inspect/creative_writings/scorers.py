"""Judge Creative Writings with the shared reasoning-inclusive trajectory scorer."""

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from cheatbench_inspect.config import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_REASONING_EFFORT
from cheatbench_inspect.creative_writings.resources.judge_schema import (
    EXTRA_CONTEXT,
    CreativeWritingsVerdict,
)
from cheatbench_inspect.sandbox_task import trajectory_scorer


@scorer(metrics=[mean()])
def creative_writings_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(
            role="grader",
            default=DEFAULT_JUDGE_MODEL,
            config=GenerateConfig(reasoning_effort=DEFAULT_JUDGE_REASONING_EFFORT),
        )
        return await trajectory_scorer(
            CreativeWritingsVerdict, EXTRA_CONTEXT, judge, explanation_field="summary"
        )(state, target)

    return score
