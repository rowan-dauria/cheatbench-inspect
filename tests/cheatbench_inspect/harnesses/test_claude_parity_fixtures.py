"""Check the recorded Claude bridge fixtures are usable without Docker."""

import json
from pathlib import Path


def test_claude_parity_fixtures() -> None:
    fixtures = Path(__file__).parent / "fixtures/claude_parity"
    request = json.loads((fixtures / "first_request.json").read_text())
    assert any(message["role"] == "system" for message in request["messages"])
    assert {"Bash", "Read", "Write", "mcp__verifier__verify_factors"} <= {
        tool["name"] for tool in request["tools"]
    }
