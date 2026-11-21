"""Default settings shared by CheatBench tasks."""

import os

# Original CheatBench core/judge.py and the gpt-6-sol profile in configs/models.yaml.
DEFAULT_JUDGE_MODEL = "openai/gpt-6-sol"
DEFAULT_JUDGE_REASONING_EFFORT = "high"

# The same judge, reached through OpenRouter.
OPENROUTER_JUDGE_MODEL = "openrouter/openai/gpt-6-sol"


def default_judge_model(evaluated_model: str) -> str:
    """Pick the judge used when no `grader` model role is given.

    The judge goes through OpenRouter when the evaluated model does, or when an
    OpenRouter key is set and an OpenAI key is not. A run set up for OpenRouter
    then needs no OpenAI key. An explicit `--model-role grader=...` still wins.
    """
    on_openrouter = evaluated_model.startswith("openrouter/")
    only_openrouter_key = bool(os.environ.get("OPENROUTER_API_KEY")) and not os.environ.get(
        "OPENAI_API_KEY"
    )
    return OPENROUTER_JUDGE_MODEL if on_openrouter or only_openrouter_key else DEFAULT_JUDGE_MODEL
