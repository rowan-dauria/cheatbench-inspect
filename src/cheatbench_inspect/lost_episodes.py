# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/errors.py
# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/judge.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

"""Classify episode failure text and identify lost infrastructure episodes.

Ported from CheatBench's regex error taxonomy (`core/agents/errors.py`) and
judgeability gate (`core/judge.py:is_judgeable`), shared across harnesses.
"""

import re
from collections.abc import Sequence

# Exact wording emitted when a runner kills the episode at its wall clock.
# https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/errors.py#L357
TIMEOUT_WORDING = r"timeout after \d+s"

# Exact turn-cap string that is never classified as lost, matching the exemption
# in original core/judge.py:175-178.
MAX_TURNS_EXACT_ERROR = "max_turns reached before confirmed task completion"

# Whether each text-classified reason tag marks an episode the harness lost
# (an infrastructure or budget failure, rather than a finding about the model).
# https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/errors.py#L98-L312
# https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/errors.py#L550-L575
#
# Programmatic-only reasons with no text patterns in original errors.py are
# omitted here: `unclassified` (L55-L87), `api_error` (L92-L95), `harness`
# (L259-L262), `container_launch` (L265-L268), `setup_hook` (L271-L274),
# `install_failed` (L277-L280), `grader_unreachable` (L314-L318),
# `reviewer_infrastructure` (L321-L325), and `batch_cancelled` (L328-L332).
# Harness-specific pattern extensions for Kimi Code (kimi_code.py#L143-L158),
# Muse Code (muse_code.py#L264-L278), and Grok Build (grok_build.py#L143-L146)
# are deferred until those harnesses are ported.
REASON_LOST: dict[str, bool] = {
    "rate_limit": True,
    "usage_limit": True,
    "api_internal_server": True,
    "overloaded": True,
    "proxy_5xx": True,
    "connection_closed": True,
    "response_stalled": True,
    "output_token_exceeded": False,
    "context_window_exceeded": False,
    "safety_refusal": False,
    "request_too_large": True,
    "invalid_api_input": True,
    "unknown_api_error": False,
    "auth": False,
    "model_not_found": False,
    "network": True,
    "timeout": False,
    "max_turns": False,
    "container_exit": False,
    "no_record": False,
    "unparseable_record": False,
}

# Verbatim regex patterns from original ERROR_PATTERNS, preserving exact strings,
# ordering, and case-insensitive matching.
# https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/errors.py#L362-L473
ORIGINAL_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"No endpoints found that support image input|does not support (?:image|vision)|image input.*not supported|Failed to deserialize the JSON body|unknown variant .?image_url",
        "invalid_api_input",
    ),
    (r"request_too_large|Request exceeds the maximum size", "request_too_large"),
    (r"rate.?limit", "rate_limit"),
    (r"too many requests", "rate_limit"),
    (r"specified API usage limits", "usage_limit"),
    (r"You've hit your usage limit", "usage_limit"),
    (r"You have an unpaid invoice", "usage_limit"),
    (r"Quota exceeded.", "usage_limit"),
    (r"Budget has been exceeded", "usage_limit"),
    (r"API Error: 500 Internal server error", "api_internal_server"),
    (r"RetriableError: \[internal\] Error", "api_internal_server"),
    (r"API Error: Overloaded", "overloaded"),
    (r"ServiceUnavailableError", "overloaded"),
    (r"Selected model is at capacity\. Please try a different model\.", "overloaded"),
    (r"(?:litellm\.)?InternalServerError:.*Server disconnected", "connection_closed"),
    (r"gateway failed: IncompleteRead:", "connection_closed"),
    (r"API Error: Connection closed mid-response", "connection_closed"),
    (r"API Error: Connection lost mid-response", "connection_closed"),
    (r"API Error: Server error mid-response", "api_internal_server"),
    (r"Repeated 529 Overloaded errors", "overloaded"),
    (r"Reached maximum number of turns", "max_turns"),
    (r"^max_turns reached before confirmed task completion$", "max_turns"),
    (r"Maximum session turns exceeded|MAX_TURNS_EXCEEDED", "max_turns"),
    (r"API Error: stream closed before completion", "connection_closed"),
    (r"API Error: Response stalled mid-stream", "response_stalled"),
    (r"response exceeded .+ output token maximum", "output_token_exceeded"),
    (
        (
            r"input token count exceeds the maximum number of tokens|"
            r"prompt is too long: \d+ tokens > \d+ maximum"
        ),
        "context_window_exceeded",
    ),
    (r"Not logged in", "auth"),
    (r"Cannot use this model", "model_not_found"),
    (
        (
            r"cloudflare-5xx-errors|"
            r"/error-50[234]/|"
            r'"cloudflare_error"\s*:\s*true|'
            r"API Error: 50[234]\b|"
            r"HTTP 50[234]\b|"
            r"\b50[234] (?:Bad Gateway|Gateway Time-?out)|"
            r"API Error: 52[0-7]\b|"
            r"HTTP 52[0-7]\b|"
            r"\b52[0-7] status code\b|"
            r"Bad Gateway|"
            r"Gateway Time-?out"
        ),
        "proxy_5xx",
    ),
    (
        (
            r"safety measures that flagged|Cyber Verification Program|"
            r"flagged for possible cybersecurity risk|"
            r"Trusted Access for Cyber|chatgpt\.com/cyber|"
            r"Output blocked by content filtering policy|"
            r"violate our Usage Policy|"
            r"https://www\.anthropic\.com/legal/aup|"
            r"triggered cyber-related safeguards|"
            r"model_refusal_no_fallback|"
            r'ContentFilterError|"reason"\s*:\s*"content-filter"'
        ),
        "safety_refusal",
    ),
    (r"API Error", "unknown_api_error"),
    (r"SSL_ERROR_SYSCALL", "network"),
    (r"SSL_connect", "network"),
    (r"Could not resolve host", "network"),
    (r"Connection refused", "network"),
    (r"Connection timed out", "network"),
    (r"Request timed out", "network"),
    (r"curl: \(\d+\)", "network"),
    (TIMEOUT_WORDING, "timeout"),
    (r"container exited -?\d+", "container_exit"),
    (r"container produced no record on stdout", "no_record"),
    (r"unparseable container stdout", "unparseable_record"),
)

# PORT-ONLY PATTERNS
#
# Inspect's `sandbox_agent_bridge` converts status-less upstream and Inspect
# exceptions (`status_code_of(ex) is None`) into HTTP 400 `invalid_request_error`,
# which Harbor's LiteLLM records as `BadRequestError: litellm.BadRequestError:
# OpenAIException - <message>`. The stock CLI may also replace provider errors
# with its own wording. Each pattern below stands in for a specific lost
# failure mode where the bridge/Inspect wording prevents ORIGINAL_PATTERNS from
# matching the category the original would assign.
PORT_ONLY_PATTERNS: tuple[tuple[str, str], ...] = (
    # The stock Codex CLI reports high demand for a Responses 500. The
    # model exception retained by the bridge has the original status code.
    (r"\bInternalServerError: Error code: 500\b", "api_internal_server"),
    # Failure mode: streaming response stalled mid-stream (Inspect StreamIdleTimeoutError).
    # Exact port message: "BadRequestError: litellm.BadRequestError: OpenAIException - stream_idle_timeout '120' exceeded (streaming response stalled)."
    # Stands in for original reason: `response_stalled` (ApiResponseStalledError, errors.py#L403).
    (r"stream_idle_timeout '\d+' exceeded \(streaming response stalled\)\.", "response_stalled"),
    # Failure mode: per-attempt model request timed out (Inspect AttemptTimeoutError).
    # Exact port message: "BadRequestError: litellm.BadRequestError: OpenAIException - attempt_timeout '300' exceeded."
    # Stands in for original reason: `network` (NetworkConnectionError, errors.py#L465).
    (r"attempt_timeout '\d+' exceeded\.", "network"),
    # Failure mode: HTTP 200 streaming response ended with zero chunks (Inspect NoStreamDataError).
    # Exact port message: "BadRequestError: litellm.BadRequestError: OpenAIException - Streaming response ended without delivering any chunks."
    # Stands in for original reason: `connection_closed` (ApiConnectionClosedError, errors.py#L401).
    (
        r"Streaming response ended without delivering any chunks\.|No (?:response|data) chunks received from streaming",
        "connection_closed",
    ),
    # Failure mode: upstream closed the connection without sending a response (httpx.RemoteProtocolError).
    # Bridge rewrites status-less error to HTTP 400 BadRequestError instead of InternalServerError.
    # Exact port message: "BadRequestError: litellm.BadRequestError: OpenAIException - Server disconnected without sending a response."
    # Stands in for original reason: `connection_closed` (ApiConnectionClosedError, errors.py#L384, test_agent_errors.py#L506).
    (r"(?:litellm\.)?BadRequestError:.*Server disconnected", "connection_closed"),
    # Failure mode: upstream closed a chunked HTTP response mid-body (httpx.RemoteProtocolError).
    # Exact port message: "BadRequestError: litellm.BadRequestError: OpenAIException - peer closed connection without sending complete message body (incomplete chunked read)"
    # Stands in for original reason: `connection_closed` (ApiConnectionClosedError, errors.py#L387-L389).
    (
        r"(?:litellm\.)?BadRequestError:.*peer closed connection without sending complete message body",
        "connection_closed",
    ),
    # Failure mode: TCP connection reset by peer (status-less connection error rewritten to HTTP 400).
    # Exact port message: "BadRequestError: litellm.BadRequestError: OpenAIException - Connection reset by peer"
    # Stands in for original reason: `connection_closed` (ApiConnectionClosedError, errors.py#L388-L392).
    (r"(?:litellm\.)?BadRequestError:.*Connection reset by peer", "connection_closed"),
    # Failure mode: OpenAI/Anthropic SDK APIConnectionError (DNS failure, connection refused, TLS/socket drop)
    # rewritten by bridge to HTTP 400 BadRequestError with the SDK's default "Connection error." message.
    # Exact port message: "BadRequestError: litellm.BadRequestError: OpenAIException - Connection error."
    # Stands in for original reason: `network` (NetworkConnectionError, errors.py#L460-L466).
    (r"(?:litellm\.)?BadRequestError:.*OpenAIException - Connection error\.", "network"),
    # Failure mode: upstream closed the connection without sending a response (httpx.RemoteProtocolError)
    # forwarded by Inspect's sandbox bridge as HTTP 400 OpenAI-dialect JSON to Codex 0.154.0.
    # Exact captured Codex error: 'codex ended the turn with an error: {"error": {"message": "Server disconnected without sending a response.", "type": "invalid_request_error", "param": null, "code": null}}'
    # Stands in for original reason: `connection_closed` (ApiConnectionClosedError, errors.py#L384).
    (
        r'\\?"error\\?"\s*:\s*\{\s*\\?"message\\?"\s*:\s*\\?"[^"]*Server disconnected',
        "connection_closed",
    ),
    # Failure mode: upstream closed a chunked HTTP response mid-body (httpx.RemoteProtocolError)
    # forwarded by Inspect's sandbox bridge as HTTP 400 OpenAI-dialect JSON to Codex 0.154.0.
    # Exact captured Codex error: 'codex ended the turn with an error: {"error": {"message": "peer closed connection without sending complete message body (incomplete chunked read)", "type": "invalid_request_error", "param": null, "code": null}}'
    # Stands in for original reason: `connection_closed` (ApiConnectionClosedError, errors.py#L387-L389).
    (
        r'\\?"error\\?"\s*:\s*\{\s*\\?"message\\?"\s*:\s*\\?"[^"]*peer closed connection without sending complete message body',
        "connection_closed",
    ),
    # Failure mode: TCP connection reset by peer forwarded by Inspect's sandbox bridge as
    # HTTP 400 OpenAI-dialect JSON to Codex 0.154.0.
    # Exact captured Codex error: 'codex ended the turn with an error: {"error": {"message": "Connection reset by peer", "type": "invalid_request_error", "param": null, "code": null}}'
    # Stands in for original reason: `connection_closed` (ApiConnectionClosedError, errors.py#L388-L392).
    (
        r'\\?"error\\?"\s*:\s*\{\s*\\?"message\\?"\s*:\s*\\?"[^"]*Connection reset by peer',
        "connection_closed",
    ),
    # Failure mode: OpenAI SDK APIConnectionError ("Connection error.") forwarded by Inspect's
    # sandbox bridge as HTTP 400 OpenAI-dialect JSON to Codex 0.154.0.
    # Exact captured Codex error: 'codex ended the turn with an error: {"error": {"message": "Connection error.", "type": "invalid_request_error", "param": null, "code": null}}'
    # Stands in for original reason: `network` (NetworkConnectionError, errors.py#L460-L466).
    (
        r'\\?"error\\?"\s*:\s*\{\s*\\?"message\\?"\s*:\s*\\?"Connection error\.\\?"',
        "network",
    ),
)

_COMPILED_ALL: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), reason)
    for pattern, reason in (*ORIGINAL_PATTERNS, *PORT_ONLY_PATTERNS)
)


def classify(
    text: str | None,
    *extra_texts: str | None,
    patterns: Sequence[tuple[str, str]] | None = None,
) -> str | None:
    """Classify failure text into a CheatBench reason tag, or None if unmatched.

    Follows original ``core.agents.errors.classify`` semantics: case-insensitive
    search over the joined text where the match ending furthest toward the end
    wins, earlier patterns win ties, and a soft ``model_refusal_fallback``
    divert without ``model_refusal_no_fallback`` skips ``safety_refusal``.
    """
    blob = "\n".join(t for t in (text, *extra_texts) if t)
    if not blob.strip():
        return None
    compiled = (
        tuple((re.compile(p, re.IGNORECASE), reason) for p, reason in patterns)
        if patterns is not None
        else _COMPILED_ALL
    )
    skip_refusal = "model_refusal_fallback" in blob and "model_refusal_no_fallback" not in blob
    best: tuple[int, str] | None = None
    for pattern, reason in compiled:
        if skip_refusal and reason == "safety_refusal":
            continue
        for match in pattern.finditer(blob):
            if best is None or match.end() > best[0]:
                best = (match.end(), reason)
    return best[1] if best is not None else None


def is_lost_episode(error: str) -> bool:
    """Whether an episode error means an infrastructure or budget failure ended it.

    Matches original ``core.judge.is_judgeable`` on the error string: lost when
    the classified reason has ``lost_episode=True``, with the exact ``max_turns``
    message never treated as lost.
    """
    if error.strip() == MAX_TURNS_EXACT_ERROR:
        return False
    reason = classify(error)
    return reason is not None and REASON_LOST.get(reason, False)


def is_harness_timeout(error: str | None) -> bool:
    return bool(error and re.match(r"^\s*timeout after \d+s", error, re.IGNORECASE))
