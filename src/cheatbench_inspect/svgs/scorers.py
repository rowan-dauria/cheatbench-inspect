"""Judge SVG competition behavior using the shared trajectory scorer."""

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from cheatbench_inspect.config import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_REASONING_EFFORT
from cheatbench_inspect.sandbox_task import trajectory_scorer
from cheatbench_inspect.svgs.resources.judge_schema import EXTRA_CONTEXT, SvgsVerdict


@scorer(metrics=[mean(), stderr()])
def svgs_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(
            role="grader",
            default=DEFAULT_JUDGE_MODEL,
            config=GenerateConfig(reasoning_effort=DEFAULT_JUDGE_REASONING_EFFORT),
        )
        return await trajectory_scorer(
            SvgsVerdict, EXTRA_CONTEXT, judge, explanation_field="summary"
        )(state, target)

    return score
