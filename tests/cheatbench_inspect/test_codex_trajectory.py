"""Check the Codex trajectory converter against fixtures produced by the original adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cheatbench_inspect.codex_trajectory import to_trajectory
from cheatbench_inspect.judge import render_trajectory

FIXTURES = Path(__file__).parent / "fixtures" / "codex"


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_original_fixtures(path: Path) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    trajectory = to_trajectory(
        fixture["input_record"],
        cli_version=fixture["cli_version"],
        cli_model=fixture["cli_model"],
    )
    assert trajectory == fixture["expected_trajectory"]
    assert render_trajectory(trajectory) == fixture["expected_rendered"]


def test_header_matches_original_format() -> None:
    fixture = json.loads((FIXTURES / "multi_turn_exec_reasoning.json").read_text(encoding="utf-8"))
    trajectory = to_trajectory(
        fixture["input_record"],
        cli_version=fixture["cli_version"],
        cli_model=fixture["cli_model"],
    )
    rendered = render_trajectory(trajectory)
    first_line = rendered.splitlines()[0]
    assert (
        first_line
        == "[init] agent=codex version=0.154.0 model=gpt-5.6-luna originator=codex_exec cwd=/workspace"
    )


def test_dropped_invalid_step_records_notes_and_keeps_valid_steps() -> None:
    record = {
        "messages": [
            {
                "timestamp": "2026-11-22T21:00:00.000Z",
                "type": "session_meta",
                "payload": {"id": "sess-drop", "cli_version": "0.154.0"},
            },
            {
                "timestamp": "2026-11-22T21:00:00.100Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": 12345,
                    "name": "shell",
                    "arguments": "{}",
                },
            },
            {
                "timestamp": "2026-11-22T21:00:00.200Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {"last_token_usage": {"input_tokens": 10, "output_tokens": 2}},
                },
            },
            {
                "timestamp": "2026-11-22T21:00:01.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Recovered."}],
                },
            },
        ]
    }
    trajectory = to_trajectory(record, cli_version="0.154.0", cli_model="gpt-5.6-luna")
    assert len(trajectory["steps"]) == 1
    assert trajectory["steps"][0]["step_id"] == 1
    assert trajectory["steps"][0]["message"] == "Recovered."
    assert trajectory["notes"].startswith(
        "1 rollout event(s) could not be converted to ATIF steps and are missing from this trajectory: bundled: ValidationError:"
    )
    assert "ToolCall" in trajectory["notes"]
    assert "tool_call_id" in trajectory["notes"]


def test_all_steps_dropped_falls_back_to_placeholder_with_notes() -> None:
    record = {
        "messages": [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": 999,
                    "name": "shell",
                    "arguments": "{}",
                },
            }
        ]
    }
    trajectory = to_trajectory(record, cli_version="0.154.0", cli_model="gpt-5.6-luna")
    assert trajectory["steps"] == [
        {
            "step_id": 1,
            "source": "system",
            "message": "(no rollout events were captured for this episode)",
            "extra": {"placeholder": True},
        }
    ]
    assert trajectory["notes"].startswith(
        "1 rollout event(s) could not be converted to ATIF steps and are missing from this trajectory:"
    )
