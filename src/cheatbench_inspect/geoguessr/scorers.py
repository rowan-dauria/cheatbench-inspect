"""Score GeoGuessr conversations with the shared judge."""

from pathlib import Path

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from cheatbench_inspect.config import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_REASONING_EFFORT
from cheatbench_inspect.geoguessr.resources.judge_schema import GeoGuessrVerdict, extra_context
from cheatbench_inspect.sandbox_task import trajectory_scorer


@scorer(metrics=[mean()])
def geoguessr_scorer(task_dir: Path) -> Scorer:
    """Judge GeoGuessr trajectories using the context built from task_dir/manifest.json."""
    context = extra_context(task_dir / "manifest.json")

    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(
            role="grader",
            default=DEFAULT_JUDGE_MODEL,
            config=GenerateConfig(reasoning_effort=DEFAULT_JUDGE_REASONING_EFFORT),
        )
        return await trajectory_scorer(
            GeoGuessrVerdict,
            context,
            judge,
            explanation_field="summary",
        )(state, target)

    return score
