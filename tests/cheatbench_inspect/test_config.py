"""Default judge routing for OpenAI and OpenRouter setups."""

import pytest

from cheatbench_inspect.config import default_judge_model


@pytest.mark.parametrize(
    ("evaluated_model", "keys", "expected"),
    [
        ("anthropic/claude-sonnet-5-5", {"OPENAI_API_KEY"}, "openai/gpt-6-sol"),
        ("anthropic/claude-sonnet-5-5", set(), "openai/gpt-6-sol"),
        ("anthropic/claude-sonnet-5-5", {"OPENROUTER_API_KEY"}, "openrouter/openai/gpt-6-sol"),
        (
            "anthropic/claude-sonnet-5-5",
            {"OPENAI_API_KEY", "OPENROUTER_API_KEY"},
            "openai/gpt-6-sol",
        ),
        ("openrouter/anthropic/claude-sonnet-5-5", set(), "openrouter/openai/gpt-6-sol"),
        (
            "openrouter/anthropic/claude-sonnet-5-5",
            {"OPENAI_API_KEY", "OPENROUTER_API_KEY"},
            "openrouter/openai/gpt-6-sol",
        ),
    ],
)
def test_default_judge_follows_openrouter_key_or_route(
    monkeypatch: pytest.MonkeyPatch, evaluated_model: str, keys: set[str], expected: str
) -> None:
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if key in keys:
            monkeypatch.setenv(key, "test-key")
        else:
            monkeypatch.delenv(key, raising=False)
    assert default_judge_model(evaluated_model) == expected
