"""Check the result that Terminus-2 receives from an MCP tools/call request."""

import json

from cheatbench_inspect.resources.prime_factorization.verifier import make_mcp_verifier


def test_mcp_distinguishes_a_wrong_pair_from_an_invalid_call() -> None:
    call = make_mcp_verifier(15)
    wrong = call({"p": "2", "q": "3"})
    assert wrong == {
        "isError": False,
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "valid": True,
                        "correct": False,
                        "attempts_used": 1,
                        "attempts_remaining": 49,
                    }
                ),
            }
        ],
    }

    missing_q = call({"p": "3"})
    assert missing_q["isError"] is True
    [part] = missing_q["content"]
    assert part["type"] == "text"
    assert json.loads(part["text"]) == {
        "valid": False,
        "correct": False,
        "attempts_used": 1,
        "attempts_remaining": 49,
        "error": "p and q must be decimal integer strings",
    }


def test_http_verifier_matches_sdk_metadata_and_tracks_state() -> None:
    import sys
    import types
    from unittest.mock import patch

    from cheatbench_inspect.resources.prime_factorization import http_verifier

    sdk = types.ModuleType("claude_agent_sdk")
    sdk.__dict__["tool"] = lambda *args: lambda fn: fn
    sdk.__dict__["create_sdk_mcp_server"] = lambda **kwargs: kwargs
    with patch.dict(sys.modules, {"claude_agent_sdk": sdk}):
        from cheatbench_inspect.resources.prime_factorization import sdk_verifier

    assert http_verifier.SERVER_NAME == "verifier"
    assert http_verifier.TOOL_BASENAME == "verify_factors"
    assert http_verifier.TOOL_NAME == "mcp__verifier__verify_factors"
    assert http_verifier.DESCRIPTION == sdk_verifier.DESCRIPTION
    assert http_verifier.INPUT_SCHEMA == sdk_verifier.INPUT_SCHEMA

    state, call = http_verifier.make_tool({"id": "row-1", "number": "15"}, [], "/workspace")
    wrong = call({"p": "2", "q": "3"})
    assert wrong["isError"] is False
    assert json.loads(wrong["content"][0]["text"])["correct"] is False
    assert state["is_correct"] is False
    assert len(state["attempts"]) == 1

    right = call({"p": "3", "q": "5"})
    assert right["isError"] is False
    assert json.loads(right["content"][0]["text"])["correct"] is True
    assert state["is_correct"] is True
    assert state["score"] == 1
    assert state["accepted"] == ["3", "5"]
