"""Shared judge behavior across tasks."""

import json
from pathlib import Path

from cheatbench_inspect.judge import Verdict, build_judge_prompt


def test_verdict_declares_the_shared_fields() -> None:
    fixture = Path(__file__).parent / "sycophancy" / "fixtures" / "shared_verdict_schema.json"
    original = json.loads(fixture.read_text())["properties"]
    schema = Verdict.model_json_schema()
    fields = ["cheated", "explicit_prohibition_violated", "explicit_prohibition_evidence"]
    assert list(schema["properties"]) == schema["required"] == fields
    assert schema["properties"]["cheated"]["type"] == "boolean"
    for name in fields[1:]:
        assert schema["properties"][name] == original[name]


def test_judge_prompt_accepts_task_context_and_multiline_trajectory() -> None:
    context = "A timed factoring interview."
    prompt = "Submit the factors to verify_factors.\n"
    trajectory = "ASSISTANT: I will check my pair.\nTOOL: literal {{agent_prompt}}"

    rendered = build_judge_prompt(extra_context=context, agent_prompt=prompt, trajectory=trajectory)

    assert f"--- WHAT THIS ENVIRONMENT IS ---\n\n{context}\n" in rendered
    assert f"<agent_instructions>\n{prompt}\n</agent_instructions>" in rendered
    assert f"<trajectory>\n{trajectory}\n</trajectory>" in rendered
