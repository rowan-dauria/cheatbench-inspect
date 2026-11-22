"""Generate the lost-episode classification fixture from the pinned CheatBench original.

Runs the original ``core.agents.errors.classify`` (read-only, with bytecode
writes disabled) over representative strings covering every ported pattern in
``core/agents/errors.py``, furthest-match precedence, the soft refusal-fallback
guard, the ``max_turns`` exemption string, and unmatched strings, and writes
the results to ``tests/cheatbench_inspect/fixtures/lost_episode_cases.json``.

Usage (against a local checkout of CheatBench at commit 96e05a6):

    PYTHONDONTWRITEBYTECODE=1 uv run python -B scripts/lost_episode_cases.py \
        --cheatbench /path/to/cheatbench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UPSTREAM_COMMIT = "96e05a6d61faf33be788f75dd01998066045ad02"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "cheatbench_inspect"
    / "fixtures"
    / "lost_episode_cases.json"
)

# Each tuple is (case_id, text). Every pattern in original
# core/agents/errors.py:ERROR_PATTERNS is covered at least once, along with
# furthest-match precedence, refusal-fallback guard, max_turns exact string,
# and strings that match nothing.
CASES: list[tuple[str, str]] = [
    # --- invalid_api_input (pattern 0) ---
    (
        "invalid_api_input_no_endpoints",
        "BadRequestError: litellm.BadRequestError: OpenAIException - No endpoints found that support image input",
    ),
    ("invalid_api_input_no_vision", "This model does not support vision"),
    ("invalid_api_input_no_image", "This model does not support image input"),
    ("invalid_api_input_image_not_supported", "image input is not supported for this route"),
    (
        "invalid_api_input_deserialize",
        "Failed to deserialize the JSON body into the target type: messages[2].content",
    ),
    ("invalid_api_input_unknown_variant", "unknown variant `image_url`, expected `text`"),
    # --- request_too_large (pattern 1) ---
    (
        "request_too_large_code",
        "ContextLengthExceededError: litellm.BadRequestError: OpenAIException - request_too_large",
    ),
    (
        "request_too_large_phrase",
        "BadRequestError: Request exceeds the maximum size allowed",
    ),
    # --- rate_limit (patterns 2-3) ---
    ("rate_limit_space", "API Error: 429 rate limit exceeded"),
    (
        "rate_limit_litellm_class",
        "RateLimitError: litellm.RateLimitError: RateLimitError: OpenAIException - Please slow down",
    ),
    ("rate_limit_hyphen", "Provider returned rate-limit error"),
    ("rate_limit_too_many_requests", "HTTP 429 Too Many Requests"),
    # --- usage_limit (patterns 4-8) ---
    (
        "usage_limit_specified",
        "You have reached your specified API usage limits. Try again later.",
    ),
    ("usage_limit_hit", "You've hit your usage limit for this billing cycle."),
    ("usage_limit_unpaid_invoice", "You have an unpaid invoice on this account."),
    ("usage_limit_quota_exceeded", "Quota exceeded. Please check your billing plan."),
    (
        "usage_limit_budget_exceeded_cli",
        (
            'API Error: 400 {"error":{"message":"litellm.BudgetExceededError: '
            'Budget has been exceeded! Current cost: 2000.0038, Max budget: 2000.0",'
            '"type":"budget_exceeded","code":"400"}}'
        ),
    ),
    (
        "usage_limit_budget_exceeded_harbor",
        "BadRequestError: litellm.BadRequestError: OpenAIException - Budget has been exceeded!",
    ),
    # --- api_internal_server (patterns 9-10, 18) ---
    ("api_internal_server_500", "API Error: 500 Internal server error"),
    ("api_internal_server_retriable", "RetriableError: [internal] Error processing request"),
    ("api_internal_server_mid_response", "API Error: Server error mid-response"),
    # --- overloaded (patterns 11-13, 19) ---
    ("overloaded_api_error", "API Error: Overloaded"),
    (
        "overloaded_service_unavailable",
        "ServiceUnavailableError: litellm.ServiceUnavailableError: ServiceUnavailableError: OpenAIException - Service unavailable",
    ),
    (
        "overloaded_at_capacity",
        "Selected model is at capacity. Please try a different model.",
    ),
    ("overloaded_repeated_529", "Repeated 529 Overloaded errors after retries"),
    # --- connection_closed (patterns 14-17, 23) ---
    (
        "connection_closed_litellm_disconnected",
        "InternalServerError: litellm.InternalServerError: AnthropicException - Server disconnected. Handle with litellm.InternalServerError.",
    ),
    (
        "connection_closed_incomplete_read",
        "gateway failed: IncompleteRead: IncompleteRead(0 bytes read)",
    ),
    ("connection_closed_mid_response", "API Error: Connection closed mid-response"),
    ("connection_lost_mid_response", "API Error: Connection lost mid-response"),
    ("connection_closed_stream", "API Error: stream closed before completion"),
    # --- max_turns (patterns 20-22) ---
    ("max_turns_reached_phrase", "Error: Reached maximum number of turns (150)"),
    ("max_turns_exact", "max_turns reached before confirmed task completion"),
    (
        "max_turns_exact_not_anchored",
        "prefix: max_turns reached before confirmed task completion",
    ),
    ("max_turns_session_exceeded", "Maximum session turns exceeded"),
    ("max_turns_constant", "Error code: MAX_TURNS_EXCEEDED"),
    # --- response_stalled (pattern 24) ---
    ("response_stalled_mid_stream", "API Error: Response stalled mid-stream"),
    # --- output_token_exceeded (pattern 25) ---
    ("output_token_exceeded", "response exceeded 64000 output token maximum"),
    # --- context_window_exceeded (pattern 26) ---
    (
        "context_window_input_tokens",
        "input token count exceeds the maximum number of tokens",
    ),
    (
        "context_window_prompt_too_long",
        "Error: prompt is too long: 214431 tokens > 200000 maximum",
    ),
    # --- auth (pattern 27) ---
    ("auth_not_logged_in", "Not logged in"),
    # --- model_not_found (pattern 28) ---
    ("model_not_found", "Cannot use this model"),
    # --- proxy_5xx (pattern 29) ---
    (
        "proxy_5xx_cloudflare_url",
        "https://developers.cloudflare.com/support/troubleshooting/cloudflare-errors/cloudflare-5xx-errors/",
    ),
    ("proxy_5xx_error_502_path", "See /error-502/ for details"),
    ("proxy_5xx_cloudflare_json", '{"cloudflare_error": true, "status": 502}'),
    ("proxy_5xx_api_error_503", "API Error: 503 upstream failed"),
    ("proxy_5xx_http_504", "HTTP 504 upstream failed"),
    ("proxy_5xx_502_bad_gateway", "502 Bad Gateway"),
    ("proxy_5xx_api_error_524", "API Error: 524 origin timed out"),
    ("proxy_5xx_http_522", "HTTP 522 connection timed out"),
    (
        "proxy_5xx_524_status_code",
        "APIError: litellm.APIError: APIError: OpenAIException - 524 status code (no body)",
    ),
    ("proxy_5xx_bad_gateway_phrase", "Bad Gateway"),
    (
        "proxy_5xx_gateway_timeout",
        "Timeout: litellm.Timeout: Timeout Error: OpenAIException - Gateway Timeout",
    ),
    ("proxy_5xx_gateway_time_out_hyphen", "504 Gateway Time-out"),
    # --- safety_refusal (pattern 30) ---
    ("safety_refusal_measures", "Request hit safety measures that flagged the prompt"),
    ("safety_refusal_cyber_program", "Requires Cyber Verification Program access"),
    ("safety_refusal_cyber_risk", "Prompt flagged for possible cybersecurity risk"),
    ("safety_refusal_trusted_access", "Trusted Access for Cyber required"),
    ("safety_refusal_chatgpt_cyber", "See chatgpt.com/cyber for access"),
    ("safety_refusal_content_filtering", "Output blocked by content filtering policy"),
    (
        "safety_refusal_usage_policy",
        (
            "API Error: Claude Code is unable to respond to this request, "
            "which appears to violate our Usage Policy "
            "(https://www.anthropic.com/legal/aup)."
        ),
    ),
    ("safety_refusal_aup_url", "Blocked under https://www.anthropic.com/legal/aup"),
    ("safety_refusal_cyber_safeguards", "Request triggered cyber-related safeguards"),
    ("safety_refusal_no_fallback", '{"subtype":"model_refusal_no_fallback"}'),
    ("safety_refusal_content_filter_error", "ContentFilterError: request blocked"),
    ("safety_refusal_content_filter_json", '{"reason": "content-filter"}'),
    # --- unknown_api_error (pattern 31) ---
    ("unknown_api_error", "API Error: unexpected response format"),
    # --- network (patterns 32-38) ---
    ("network_ssl_error_syscall", "OpenSSL SSL_read: SSL_ERROR_SYSCALL, errno 104"),
    ("network_ssl_connect", "OpenSSL SSL_connect: connection reset"),
    ("network_could_not_resolve", "Could not resolve host: api.example.com"),
    ("network_connection_refused", "OSError: [Errno 111] Connection refused"),
    ("network_connection_timed_out", "TCP Connection timed out after 30s"),
    (
        "network_request_timed_out",
        "Timeout: litellm.Timeout: APITimeoutError - Request timed out.",
    ),
    ("network_curl_error", "curl: (60) SSL certificate problem"),
    # --- harness runner wordings (patterns 39-42) ---
    ("timeout_runner", "timeout after 3600s"),
    ("container_exit_positive", "container exited 137"),
    ("container_exit_negative", "container exited -9"),
    ("no_record", "container produced no record on stdout"),
    ("unparseable_record", "unparseable container stdout: Expecting value"),
    # --- furthest-match precedence ---
    (
        "furthest_match_context_after_rate_limit",
        "API Error: 429 rate limit\n...retrying...\nError: prompt is too long: 214431 tokens > 200000 maximum",
    ),
    (
        "furthest_match_rate_limit_after_context",
        "Error: prompt is too long: 214431 tokens > 200000 maximum\nAPI Error: 429 rate limit",
    ),
    (
        "furthest_match_quota_after_rate_limit_class",
        "RateLimitError: litellm.RateLimitError: OpenAIException - Quota exceeded.",
    ),
    (
        "furthest_match_refusal_beats_earlier_api_error",
        "API Error: Output blocked by content filtering policy",
    ),
    (
        "furthest_match_budget_beats_earlier_api_error",
        "API Error: Budget has been exceeded",
    ),
    (
        "furthest_match_500_beats_earlier_api_error",
        "API Error: 500 Internal server error",
    ),
    # --- refusal-fallback guard ---
    (
        "refusal_guard_soft_divert_falls_back_to_api_error",
        (
            '{"subtype":"model_refusal_fallback"} API Error: Claude Code is unable to '
            "respond to this request, which appears to violate our Usage Policy "
            "(https://www.anthropic.com/legal/aup)."
        ),
    ),
    (
        "refusal_guard_soft_divert_alone_is_none",
        '{"subtype":"model_refusal_fallback"} Output blocked by content filtering policy',
    ),
    (
        "refusal_guard_hard_stop_overrides_soft_divert",
        (
            '{"subtype":"model_refusal_fallback"} {"subtype":"model_refusal_no_fallback"} '
            "Output blocked by content filtering policy"
        ),
    ),
    (
        "refusal_guard_is_case_sensitive",
        '{"subtype":"MODEL_REFUSAL_FALLBACK"} Output blocked by content filtering policy',
    ),
    # --- unmatched strings ---
    ("unmatched_empty", ""),
    ("unmatched_whitespace", "   \n\t  "),
    ("unmatched_plain_text", "the agent wrote a file and stopped"),
    ("unmatched_timeout_error_class", "TimeoutError: "),
    ("unmatched_context_length_class", "ContextLengthExceededError: "),
    (
        "unmatched_generic_bad_request",
        "BadRequestError: litellm.BadRequestError: OpenAIException - invalid parameter foo",
    ),
    (
        "unmatched_auth_error",
        "AuthenticationError: litellm.AuthenticationError: AuthenticationError: OpenAIException - Incorrect API key provided",
    ),
    (
        "unmatched_not_found",
        "NotFoundError: litellm.NotFoundError: OpenAIException - The model does not exist",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate lost_episode_cases.json from the pinned CheatBench original."
    )
    parser.add_argument(
        "--cheatbench",
        type=Path,
        required=True,
        help=f"Path to a read-only checkout of CheatBench at {UPSTREAM_COMMIT[:7]}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON fixture path.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.cheatbench.resolve()))
    from core.agents import errors as orig_errors

    records = []
    for case_id, text in CASES:
        classified = orig_errors.classify(text)
        records.append(
            {
                "id": case_id,
                "text": text,
                "reason": orig_errors.reason(classified),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} cases to {args.output}")


if __name__ == "__main__":
    main()
