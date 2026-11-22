"""Check the converter against fixtures produced by the original adapter."""

import base64
import json
from pathlib import Path

import pytest

from cheatbench_inspect.claude_trajectory import to_trajectory
from cheatbench_inspect.judge import render_trajectory

FIXTURES = Path(__file__).parent / "fixtures/claude_sdk"


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_original_fixtures(path):
    fixture = json.loads(path.read_text())
    trajectory = to_trajectory(
        fixture["record"], cli_version="2.1.270", cli_model="claude-sonnet-5"
    )
    assert trajectory == fixture["trajectory"]
    assert render_trajectory(trajectory) == fixture["rendered"]


def test_openrouter_reasoning_shapes():
    data = base64.b64encode(
        json.dumps({"text": "private thought", "type": "reasoning.text"}).encode()
    ).decode()
    envelope = "openrouter.reasoning:" + data

    def render(block):
        record = {"messages": [{"_type": "AssistantMessage", "content": [block]}]}
        return render_trajectory(
            to_trajectory(record, cli_version="2.1.270", cli_model="claude-sonnet-5")
        )

    original = render({"_type": "RedactedThinkingBlock", "data": envelope})
    port = render({"_type": "ThinkingBlock", "thinking": "private thought", "signature": envelope})
    assert original == port
