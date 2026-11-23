"""Tests for shared lost-episode error classification."""

import json
from pathlib import Path

import pytest

from cheatbench_inspect.lost_episodes import (
    MAX_TURNS_EXACT_ERROR,
    ORIGINAL_PATTERNS,
    PORT_ONLY_PATTERNS,
    REASON_LOST,
    classify,
    is_lost_episode,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "lost_episode_cases.json"
FIXTURE_CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", FIXTURE_CASES, ids=[c["id"] for c in FIXTURE_CASES])
def test_classify_matches_original_fixture(case: dict[str, str | None]) -> None:
    text = case["text"]
    assert isinstance(text, str)
    assert classify(text, patterns=ORIGINAL_PATTERNS) == case["reason"]


@pytest.mark.parametrize("case", FIXTURE_CASES, ids=[c["id"] for c in FIXTURE_CASES])
def test_port_only_patterns_do_not_change_original_fixture(case: dict[str, str | None]) -> None:
    text = case["text"]
    assert isinstance(text, str)
    assert classify(text) == case["reason"]


def test_classify_handles_none_and_multiple_texts() -> None:
    assert classify(None) is None
    assert classify(None, "") is None
    assert (
        classify(
            "API Error: 429 rate limit",
            "Error: prompt is too long: 214431 tokens > 200000 maximum",
        )
        == "context_window_exceeded"
    )


def test_earlier_pattern_wins_same_end_tie() -> None:
    patterns = (
        (r"foo bar", "first_reason"),
        (r"bar", "second_reason"),
    )
    assert classify("prefix foo bar", patterns=patterns) == "first_reason"
    assert classify("prefix foo bar", patterns=tuple(reversed(patterns))) == "second_reason"


def test_is_lost_episode_matches_original_lost_flags() -> None:
    for case in FIXTURE_CASES:
        text = case["text"]
        reason = case["reason"]
        assert isinstance(text, str)
        expected = reason is not None and REASON_LOST[reason]
        assert is_lost_episode(text) is expected, case["id"]


def test_exact_max_turns_error_is_never_lost() -> None:
    assert is_lost_episode(MAX_TURNS_EXACT_ERROR) is False
    assert is_lost_episode(f"  {MAX_TURNS_EXACT_ERROR}\n") is False


# One captured port message per entry in PORT_ONLY_PATTERNS, asserting the
# original reason it stands in for (and that it is classified as lost).
PORT_ONLY_CASES = [
    (
        "stock_codex_upstream_http_500",
        "InternalServerError: Error code: 500 - Scripted provider failure",
        "api_internal_server",
    ),
    (
        "stream_idle_timeout",
        (
            "BadRequestError: litellm.BadRequestError: OpenAIException - "
            "stream_idle_timeout '120' exceeded (streaming response stalled)."
        ),
        "response_stalled",
    ),
    (
        "attempt_timeout",
        (
            "BadRequestError: litellm.BadRequestError: OpenAIException - "
            "attempt_timeout '300' exceeded."
        ),
        "network",
    ),
    (
        "no_stream_data",
        (
            "BadRequestError: litellm.BadRequestError: OpenAIException - "
            "Streaming response ended without delivering any chunks."
        ),
        "connection_closed",
    ),
    (
        "server_disconnected_as_bad_request",
        (
            "BadRequestError: litellm.BadRequestError: OpenAIException - "
            "Server disconnected without sending a response."
        ),
        "connection_closed",
    ),
    (
        "incomplete_chunked_read_as_bad_request",
        (
            "BadRequestError: litellm.BadRequestError: OpenAIException - "
            "peer closed connection without sending complete message body (incomplete chunked read)"
        ),
        "connection_closed",
    ),
    (
        "connection_reset_by_peer_as_bad_request",
        "BadRequestError: litellm.BadRequestError: OpenAIException - Connection reset by peer",
        "connection_closed",
    ),
    (
        "sdk_connection_error_as_bad_request",
        "BadRequestError: litellm.BadRequestError: OpenAIException - Connection error.",
        "network",
    ),
    (
        "codex_server_disconnected_as_openai_400",
        (
            'codex ended the turn with an error: {"error": {"message": '
            '"Server disconnected without sending a response.", '
            '"type": "invalid_request_error", "param": null, "code": null}}'
        ),
        "connection_closed",
    ),
    (
        "codex_incomplete_chunked_read_as_openai_400",
        (
            'codex ended the turn with an error: {"error": {"message": '
            '"peer closed connection without sending complete message body '
            '(incomplete chunked read)", '
            '"type": "invalid_request_error", "param": null, "code": null}}'
        ),
        "connection_closed",
    ),
    (
        "codex_connection_reset_by_peer_as_openai_400",
        (
            'codex ended the turn with an error: {"error": {"message": '
            '"Connection reset by peer", '
            '"type": "invalid_request_error", "param": null, "code": null}}'
        ),
        "connection_closed",
    ),
    (
        "codex_sdk_connection_error_as_openai_400",
        (
            'codex ended the turn with an error: {"error": {"message": '
            '"Connection error.", '
            '"type": "invalid_request_error", "param": null, "code": null}}'
        ),
        "network",
    ),
]


def test_every_port_only_pattern_has_a_dedicated_case() -> None:
    assert len(PORT_ONLY_CASES) == len(PORT_ONLY_PATTERNS)
    for (case_id, message, expected_reason), pattern_entry in zip(
        PORT_ONLY_CASES, PORT_ONLY_PATTERNS, strict=True
    ):
        assert classify(message, patterns=(pattern_entry,)) == expected_reason, case_id


@pytest.mark.parametrize(
    ("message", "expected_reason"),
    [(msg, reason) for _, msg, reason in PORT_ONLY_CASES],
    ids=[case_id for case_id, _, _ in PORT_ONLY_CASES],
)
def test_port_only_pattern_matches_captured_port_message(
    message: str, expected_reason: str
) -> None:
    assert classify(message, patterns=ORIGINAL_PATTERNS) is None
    assert classify(message) == expected_reason
    assert is_lost_episode(message) is True


def test_bare_bad_request_is_not_lost() -> None:
    message = "BadRequestError: litellm.BadRequestError: OpenAIException - invalid parameter"
    assert classify(message) is None
    assert is_lost_episode(message) is False
    codex_bad_request = (
        'codex ended the turn with an error: {"error": {"message": "invalid parameter", '
        '"type": "invalid_request_error", "param": null, "code": null}}'
    )
    assert classify(codex_bad_request) is None
    assert is_lost_episode(codex_bad_request) is False


@pytest.mark.parametrize(
    ("message", "expected_reason"),
    [
        (
            (
                'codex ended the turn with an error: {"error": {"message": '
                "\"stream_idle_timeout '120' exceeded (streaming response stalled).\", "
                '"type": "invalid_request_error", "param": null, "code": null}}'
            ),
            "response_stalled",
        ),
        (
            (
                'codex ended the turn with an error: {"error": {"message": '
                "\"attempt_timeout '300' exceeded.\", "
                '"type": "invalid_request_error", "param": null, "code": null}}'
            ),
            "network",
        ),
        (
            (
                'codex ended the turn with an error: {"error": {"message": '
                '"Streaming response ended without delivering any chunks.", '
                '"type": "invalid_request_error", "param": null, "code": null}}'
            ),
            "connection_closed",
        ),
        (
            (
                '{"type":"turn.failed","error":{"message":"{\\"error\\": {\\"message\\": '
                '\\"Server disconnected without sending a response.\\", \\"type\\": '
                '\\"invalid_request_error\\", \\"param\\": null, \\"code\\": null}}"}}'
            ),
            "connection_closed",
        ),
        (
            (
                '{"type":"turn.failed","error":{"message":"{\\"error\\": {\\"message\\": '
                '\\"Connection error.\\", \\"type\\": \\"invalid_request_error\\", '
                '\\"param\\": null, \\"code\\": null}}"}}'
            ),
            "network",
        ),
    ],
)
def test_captured_codex_bridge_errors_classify_as_lost(message: str, expected_reason: str) -> None:
    assert classify(message) == expected_reason
    assert is_lost_episode(message) is True
