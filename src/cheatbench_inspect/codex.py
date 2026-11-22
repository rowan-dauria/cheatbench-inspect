"""Codex CLI harness failure classification (launcher module)."""

from __future__ import annotations

from typing import Any

from cheatbench_inspect.lost_episodes import (
    MAX_TURNS_EXACT_ERROR,
    REASON_LOST,
    classify,
    is_harness_timeout,
)

# Matches original core/trial.py:1007 (`CLEAN_EXIT_SCAN_LINES`).
_CLEAN_EXIT_SCAN_LINES = 20


def classify_codex_failure(record: dict[str, Any], stderr_tail: str) -> tuple[str | None, bool]:
    """Apply the original trial's failure precedence to a native Codex record.

    Mirrors ``core/trial.py:930-999`` (``classify_failure``) and
    ``core/trial.py:1046-1093`` (``lost_on_a_clean_exit``) as well as
    ``cheatbench_inspect.claude_sdk.classify_claude_failure``.
    """
    error = record.get("error")
    if error:
        if (
            isinstance(record.get("install"), dict) and record["install"].get("status") == "failed"
        ) or str(error).startswith("codex install failed:"):
            return error, False
        if str(error).startswith(MAX_TURNS_EXACT_ERROR) or is_harness_timeout(error):
            return error, False
        reason = classify(error, stderr_tail)
        return error, REASON_LOST.get(reason or "", False)
    if not (record.get("final_text") or "").strip():
        lines = [line for line in stderr_tail.splitlines() if line.strip()][
            -_CLEAN_EXIT_SCAN_LINES:
        ]
        if lines:
            reason = classify("\n".join(lines))
            if REASON_LOST.get(reason or "", False):
                return f"codex exited 0 after: {lines[-1].strip()}", True
    return None, False
