"""Score Synthetic Puzzles conversations with the shared judge."""

from pathlib import Path

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from cheatbench_inspect.config import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_REASONING_EFFORT
from cheatbench_inspect.sandbox_task import trajectory_scorer
from cheatbench_inspect.synthetic_puzzles.resources.judge_schema import (
    SyntheticPuzzlesVerdict,
    extra_context,
)


@scorer(metrics=[mean()])
def synthetic_puzzles_scorer(task_dir: Path) -> Scorer:
    """Judge Synthetic Puzzles trajectories using the context built from task_dir/data.jsonl."""
    context = extra_context(task_dir / "data.jsonl")

    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(
            role="grader",
            default=DEFAULT_JUDGE_MODEL,
            config=GenerateConfig(reasoning_effort=DEFAULT_JUDGE_REASONING_EFFORT),
        )
        return await trajectory_scorer(
            SyntheticPuzzlesVerdict,
            context,
            judge,
            explanation_field="summary",
        )(state, target)

    return score
