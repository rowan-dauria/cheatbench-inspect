"""Failure classification precedence for the Codex CLI harness."""

from __future__ import annotations

import pytest

from cheatbench_inspect.codex import classify_codex_failure

_CAPTURED_TRANSPORT_ERRORS = [
    (
        "stream_idle_timeout",
        (
            'codex ended the turn with an error: {"error": {"message": '
            "\"stream_idle_timeout '120' exceeded (streaming response stalled).\", "
            '"type": "invalid_request_error", "param": null, "code": null}}'
        ),
    ),
    (
        "attempt_timeout",
        (
            'codex ended the turn with an error: {"error": {"message": '
            "\"attempt_timeout '300' exceeded.\", "
            '"type": "invalid_request_error", "param": null, "code": null}}'
        ),
    ),
    (
        "no_stream_data",
        (
            'codex ended the turn with an error: {"error": {"message": '
            '"Streaming response ended without delivering any chunks.", '
            '"type": "invalid_request_error", "param": null, "code": null}}'
        ),
    ),
    (
        "server_disconnected",
        (
            'codex ended the turn with an error: {"error": {"message": '
            '"Server disconnected without sending a response.", '
            '"type": "invalid_request_error", "param": null, "code": null}}'
        ),
    ),
    (
        "incomplete_chunked_read",
        (
            'codex ended the turn with an error: {"error": {"message": '
            '"peer closed connection without sending complete message body '
            '(incomplete chunked read)", '
            '"type": "invalid_request_error", "param": null, "code": null}}'
        ),
    ),
    (
        "connection_reset_by_peer",
        (
            'codex ended the turn with an error: {"error": {"message": '
            '"Connection reset by peer", '
            '"type": "invalid_request_error", "param": null, "code": null}}'
        ),
    ),
    (
        "sdk_connection_error",
        (
            'codex ended the turn with an error: {"error": {"message": '
            '"Connection error.", '
            '"type": "invalid_request_error", "param": null, "code": null}}'
        ),
    ),
]


@pytest.mark.parametrize(
    ("record", "stderr", "expected_error", "expected_lost"),
    [
        # Install failure outranks transient stderr.
        (
            {"error": "install failed", "install": {"status": "failed"}},
            "502 Bad Gateway",
            "install failed",
            False,
        ),
        (
            {"error": "codex install failed: 502 Bad Gateway"},
            "502 Bad Gateway",
            "codex install failed: 502 Bad Gateway",
            False,
        ),
        # Harness timeout outranks transient stderr.
        (
            {"error": "timeout after 30s"},
            "502 Bad Gateway",
            "timeout after 30s",
            False,
        ),
        # Exact max_turns error is never lost.
        (
            {"error": "max_turns reached before confirmed task completion"},
            "502 Bad Gateway",
            "max_turns reached before confirmed task completion",
            False,
        ),
        # Context-length error is a finding, not lost.
        (
            {
                "error": (
                    'codex ended the turn with an error: {"error": {"message": '
                    '"Your input exceeds the context window of this model. '
                    'Please adjust your input and try again.", '
                    '"type": "invalid_request_error", "param": null, "code": null}}'
                )
            },
            "",
            (
                'codex ended the turn with an error: {"error": {"message": '
                '"Your input exceeds the context window of this model. '
                'Please adjust your input and try again.", '
                '"type": "invalid_request_error", "param": null, "code": null}}'
            ),
            False,
        ),
        (
            {
                "error": (
                    "codex ended the turn with an error: input token count exceeds "
                    "the maximum number of tokens"
                )
            },
            "",
            (
                "codex ended the turn with an error: input token count exceeds "
                "the maximum number of tokens"
            ),
            False,
        ),
        # Bare 400 client error is not lost.
        (
            {
                "error": (
                    'codex ended the turn with an error: {"error": {"message": '
                    '"invalid parameter", "type": "invalid_request_error", '
                    '"param": null, "code": null}}'
                )
            },
            "",
            (
                'codex ended the turn with an error: {"error": {"message": '
                '"invalid parameter", "type": "invalid_request_error", '
                '"param": null, "code": null}}'
            ),
            False,
        ),
        # Empty rollout with rate-limit stderr -> lost.
        (
            {"error": "codex exited 1 and wrote no rollout", "messages": []},
            "exceeded retry limit, last status: 429 Too Many Requests",
            "codex exited 1 and wrote no rollout",
            True,
        ),
        # Clean exit with empty final text and lost stderr tail -> lost.
        (
            {"error": None, "final_text": ""},
            "starting...\n502 Bad Gateway\n",
            "codex exited 0 after: 502 Bad Gateway",
            True,
        ),
        # Clean exit with empty final text and Codex JSON-escaped transport error on stderr -> lost.
        (
            {"error": None, "final_text": "   "},
            (
                '{"type":"turn.failed","error":{"message":"{\\"error\\": '
                '{\\"message\\": \\"Connection error.\\", \\"type\\": '
                '\\"invalid_request_error\\", \\"param\\": null, \\"code\\": null}}"}}\n'
            ),
            (
                'codex exited 0 after: {"type":"turn.failed","error":{"message":"{\\"error\\": '
                '{\\"message\\": \\"Connection error.\\", \\"type\\": '
                '\\"invalid_request_error\\", \\"param\\": null, \\"code\\": null}}"}}'
            ),
            True,
        ),
        # Clean exit with non-empty final text ignores lost stderr tail.
        (
            {"error": None, "final_text": "Factors are 3 and 5."},
            "502 Bad Gateway\n",
            None,
            False,
        ),
        # Clean exit with empty final text and non-lost stderr stays clean.
        (
            {"error": None, "final_text": ""},
            "input token count exceeds the maximum number of tokens\n",
            None,
            False,
        ),
    ],
)
def test_classify_codex_failure_precedence(
    record: dict, stderr: str, expected_error: str | None, expected_lost: bool
) -> None:
    assert classify_codex_failure(record, stderr) == (expected_error, expected_lost)


@pytest.mark.parametrize(
    ("case_id", "error"),
    _CAPTURED_TRANSPORT_ERRORS,
    ids=[cid for cid, _ in _CAPTURED_TRANSPORT_ERRORS],
)
def test_task_complete_captured_transport_errors_are_lost(case_id: str, error: str) -> None:
    del case_id
    assert classify_codex_failure({"error": error}, "") == (error, True)
