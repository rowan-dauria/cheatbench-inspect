"""Generate Codex trajectory parity fixtures from original CheatBench 96e05a6.

Runs the original ``core.agents.codex.CodexAgent.to_trajectory`` and
``core.render.render_trajectory`` in a read-only subprocess with
``PYTHONDONTWRITEBYTECODE=1``, ``python -B``, and a scratch working directory so
the original checkout is never written to.

Usage:

    uv run python scripts/codex_trajectory_fixtures.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "96e05a6d61faf33be788f75dd01998066045ad02"
DEFAULT_CHEATBENCH = Path("/home/agent/cheatbench")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "cheatbench_inspect" / "fixtures" / "codex"
)

_EVAL_SCRIPT = textwrap.dedent(
    """\
    import json
    import sys

    sys.path.insert(0, sys.argv[1])
    from core.agents.codex import CodexAgent
    from core.render import render_trajectory

    cases = json.loads(sys.stdin.read())
    out = {}
    for name, spec in cases.items():
        agent = object.__new__(CodexAgent)
        agent.model = spec["cli_model"]
        cli_version = spec["cli_version"]
        agent.version = lambda v=cli_version: v
        traj = agent.to_trajectory(spec["input_record"])
        out[name] = {
            "input_record": spec["input_record"],
            "cli_version": spec["cli_version"],
            "cli_model": spec["cli_model"],
            "expected_trajectory": traj.to_json_dict(),
            "expected_rendered": render_trajectory(traj),
        }
    sys.stdout.write(json.dumps(out))
    """
)


def _session_meta(
    *,
    session_id: str,
    timestamp: str,
    cwd: str = "/workspace",
    originator: str | None = "codex_exec",
    cli_version: str | None = "0.154.0",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "id": session_id,
        "timestamp": timestamp,
        "cwd": cwd,
        "source": "exec",
        "thread_source": "user",
        "model_provider": "bridge",
        "base_instructions": {"text": "You are Codex, an agent based on GPT-5."},
        "history_mode": "paginated",
        "context_window": {"window_id": f"win-{session_id}"},
    }
    if originator is not None:
        payload["originator"] = originator
    if cli_version is not None:
        payload["cli_version"] = cli_version
    return {
        "timestamp": timestamp,
        "ordinal": 0,
        "type": "session_meta",
        "payload": payload,
    }


def _turn_context(
    *,
    timestamp: str,
    ordinal: int,
    turn_id: str,
    model: str,
    cwd: str = "/workspace",
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "ordinal": ordinal,
        "type": "turn_context",
        "payload": {
            "turn_id": turn_id,
            "root_turn_id": turn_id,
            "cwd": cwd,
            "workspace_roots": [cwd],
            "current_date": "2026-11-22",
            "timezone": "Etc/UTC",
            "approval_policy": "never",
            "approvals_reviewer": "user",
            "sandbox_policy": {"type": "danger-full-access"},
            "permission_profile": {"type": "disabled"},
            "model": model,
            "personality": "pragmatic",
            "collaboration_mode": {
                "mode": "default",
                "settings": {
                    "model": model,
                    "reasoning_effort": "high",
                    "developer_instructions": None,
                },
            },
            "summary": "auto",
        },
    }


def _build_cases() -> dict[str, dict[str, Any]]:
    # 1. Realistic multi-turn custom_tool_call (`exec` JS input, list-of-input_text output)
    # with native reasoning summaries, developer/user messages, world_state/item_completed/
    # token_usage_record noise events, and per-turn + final token_count metrics.
    multi_turn_exec: dict[str, Any] = {
        "cli_version": "0.154.0",
        "cli_model": "openai/gpt-5.6-luna",
        "input_record": {
            "id": "pf-15",
            "model": "openai/gpt-5.6-luna",
            "session_id": "01a2051e-6f70-7000-8000-000000000001",
            "final_text": "Factors of 15 are 3 and 5.",
            "messages": [
                _session_meta(
                    session_id="01a2051e-6f70-7000-8000-000000000001",
                    timestamp="2026-11-22T21:16:50.450Z",
                ),
                {
                    "timestamp": "2026-11-22T21:16:50.460Z",
                    "ordinal": 1,
                    "type": "event_msg",
                    "payload": {
                        "type": "task_started",
                        "turn_id": "01a2051e-6f90-7b92-aede-ce4677bd58fd",
                        "started_at": 1795382210,
                        "model_context_window": 258400,
                        "collaboration_mode_kind": "default",
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:50.470Z",
                    "ordinal": 2,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "msg_dev_01",
                        "role": "developer",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "<permissions instructions>\nApproval policy is never.\n</permissions instructions>",
                            },
                            {
                                "type": "input_text",
                                "text": "\nSandbox mode is danger-full-access.",
                            },
                        ],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:50.479Z",
                    "ordinal": 3,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "msg_env_01",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "<environment_context>\n  <cwd>/workspace</cwd>\n  <shell>bash</shell>\n</environment_context>",
                            }
                        ],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:50.482Z",
                    "ordinal": 4,
                    "type": "world_state",
                    "payload": {"full": True, "state": {"model": "gpt-5.6-luna"}},
                },
                _turn_context(
                    timestamp="2026-11-22T21:16:50.485Z",
                    ordinal=5,
                    turn_id="01a2051e-6f90-7b92-aede-ce4677bd58fd",
                    model="gpt-5.6-luna",
                ),
                {
                    "timestamp": "2026-11-22T21:16:50.493Z",
                    "ordinal": 6,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "msg_user_01",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Factor 15 and verify via MCP.",
                            }
                        ],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:50.495Z",
                    "ordinal": 7,
                    "type": "event_msg",
                    "payload": {
                        "type": "item_completed",
                        "thread_id": "01a2051e-6f70-7000-8000-000000000001",
                        "turn_id": "01a2051e-6f90-7b92-aede-ce4677bd58fd",
                        "item": {"type": "UserMessage", "id": "u1", "content": []},
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:51.627Z",
                    "ordinal": 8,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "id": "rs_sum_turn1_aa",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "**Planning command execution**\n\nI will check 3 * 5 and call the verifier.",
                            }
                        ],
                        "content": None,
                        "encrypted_content": "gAAAAABrSum1_FAKE_ENCRYPTED_PAYLOAD==",
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:51.630Z",
                    "ordinal": 9,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "msg_asst_01",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Checking the factors with exec.",
                            }
                        ],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:51.631Z",
                    "ordinal": 10,
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "id": "ctc_01",
                        "status": "completed",
                        "call_id": "call_exec_001",
                        "name": "exec",
                        "namespace": "functions",
                        "input": 'const r = await tools.exec_command({cmd:"python3 -c \'print(3*5)\'",workdir:"/workspace",yield_time_ms:10000}); text(r.output);\n',
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:51.633Z",
                    "ordinal": 11,
                    "type": "token_usage_record",
                    "payload": {
                        "turn_id": "01a2051e-6f90-7b92-aede-ce4677bd58fd",
                        "usage": {"input_tokens": 1000, "output_tokens": 80},
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:51.697Z",
                    "ordinal": 12,
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "id": "ctco_01",
                        "call_id": "call_exec_001",
                        "output": [
                            {
                                "type": "input_text",
                                "text": "Script completed\nWall time 0.1 seconds\nOutput:\n",
                            },
                            {"type": "input_text", "text": "15\n"},
                        ],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:51.700Z",
                    "ordinal": 13,
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 1000,
                                "cached_input_tokens": 200,
                                "cache_write_input_tokens": 150,
                                "output_tokens": 80,
                                "reasoning_output_tokens": 30,
                                "total_tokens": 1080,
                            },
                            "last_token_usage": {
                                "input_tokens": 1000,
                                "cached_input_tokens": 200,
                                "cache_write_input_tokens": 150,
                                "output_tokens": 80,
                                "reasoning_output_tokens": 30,
                                "total_tokens": 1080,
                            },
                            "model_context_window": 258400,
                        },
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:53.734Z",
                    "ordinal": 14,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "id": "rs_sum_turn2_bb",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "**Reporting output**\n\nThe product matches 15.",
                            }
                        ],
                        "content": None,
                        "encrypted_content": "gAAAAABrSum2_FAKE_ENCRYPTED_PAYLOAD==",
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:53.735Z",
                    "ordinal": 15,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "msg_asst_02",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Factors of 15 are 3 and 5.",
                            }
                        ],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:53.740Z",
                    "ordinal": 16,
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 2100,
                                "cached_input_tokens": 1100,
                                "cache_write_input_tokens": 250,
                                "output_tokens": 130,
                                "reasoning_output_tokens": 55,
                                "total_tokens": 2230,
                            },
                            "last_token_usage": {
                                "input_tokens": 1100,
                                "cached_input_tokens": 900,
                                "cache_write_input_tokens": 100,
                                "output_tokens": 50,
                                "reasoning_output_tokens": 25,
                                "total_tokens": 1150,
                            },
                            "model_context_window": 258400,
                        },
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:53.745Z",
                    "ordinal": 17,
                    "type": "event_msg",
                    "payload": {
                        "type": "task_complete",
                        "turn_id": "01a2051e-6f90-7b92-aede-ce4677bd58fd",
                        "last_agent_message": "Factors of 15 are 3 and 5.",
                        "duration_ms": 3295,
                    },
                },
            ],
        },
    }

    # 2. Encrypted-only reasoning (empty summary) followed by text, plus a second
    # turn where encrypted-only reasoning follows a summary and must not clear it.
    encrypted_only: dict[str, Any] = {
        "cli_version": "0.154.0",
        "cli_model": "openai/gpt-5.6-luna",
        "input_record": {
            "id": "enc-only",
            "model": "openai/gpt-5.6-luna",
            "session_id": "01a2051d-d602-7442-821f-9395fc0aa379",
            "final_text": "smoke-ok-4721",
            "messages": [
                _session_meta(
                    session_id="01a2051d-d602-7442-821f-9395fc0aa379",
                    timestamp="2026-11-22T21:16:11.138Z",
                ),
                {
                    "timestamp": "2026-11-22T21:16:11.172Z",
                    "ordinal": 1,
                    "type": "event_msg",
                    "payload": {
                        "type": "task_started",
                        "turn_id": "turn-enc-1",
                    },
                },
                _turn_context(
                    timestamp="2026-11-22T21:16:11.195Z",
                    ordinal=2,
                    turn_id="turn-enc-1",
                    model="openai/gpt-5.6-luna",
                ),
                {
                    "timestamp": "2026-11-22T21:16:11.205Z",
                    "ordinal": 3,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Echo smoke-ok-4721."}],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:12.345Z",
                    "ordinal": 4,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "id": "rs_enc_only_1",
                        "summary": [],
                        "content": None,
                        "encrypted_content": "gAAAAABrAxRY_ENCRYPTED_ONLY_BLOB_NEVER_RENDERED==",
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:12.348Z",
                    "ordinal": 5,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Running the command now."}],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:12.350Z",
                    "ordinal": 6,
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 0,
                                "cached_input_tokens": 0,
                                "output_tokens": 20,
                            },
                            "last_token_usage": {
                                "input_tokens": 0,
                                "cached_input_tokens": 0,
                                "output_tokens": 20,
                            },
                        },
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:13.001Z",
                    "ordinal": 7,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "id": "rs_sum_kept",
                        "summary": [
                            {"type": "summary_text", "text": "Summary before encrypted block."}
                        ],
                        "content": None,
                        "encrypted_content": "gAAAAABrKeep1==",
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:13.002Z",
                    "ordinal": 8,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "id": "rs_enc_after_sum",
                        "summary": [],
                        "content": None,
                        "encrypted_content": "gAAAAABrEmptyAfterSummary==",
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:13.005Z",
                    "ordinal": 9,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "smoke-ok-4721"}],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:16:13.010Z",
                    "ordinal": 10,
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 500,
                                "cached_input_tokens": 0,
                                "output_tokens": 35,
                                "reasoning_output_tokens": 10,
                                "total_tokens": 535,
                            },
                            "last_token_usage": {
                                "input_tokens": 500,
                                "cached_input_tokens": 0,
                                "output_tokens": 15,
                                "reasoning_output_tokens": 10,
                                "total_tokens": 515,
                            },
                        },
                    },
                },
            ],
        },
    }

    # 3. Consecutive reasoning items, multiple messages + out-of-order tool outputs
    # bundled in a single api_call_id, and total_cost in token_count.
    consecutive_reasoning: dict[str, Any] = {
        "cli_version": "0.154.0",
        "cli_model": "openai/gpt-5.6-luna",
        "input_record": {
            "id": "consec-reason",
            "model": "openai/gpt-5.6-luna",
            "messages": [
                _session_meta(
                    session_id="01a2051f-0b88-7a00-8a72-82016fd23908",
                    timestamp="2026-11-22T21:17:30.400Z",
                ),
                {
                    "timestamp": "2026-11-22T21:17:30.410Z",
                    "ordinal": 1,
                    "type": "event_msg",
                    "payload": {
                        "type": "task_started",
                        "turn_id": "01a2051f-0baf-7bc0-9d0a-076deb934378",
                    },
                },
                _turn_context(
                    timestamp="2026-11-22T21:17:30.440Z",
                    ordinal=2,
                    turn_id="01a2051f-0baf-7bc0-9d0a-076deb934378",
                    model="openai/gpt-5.6-luna",
                ),
                {
                    "timestamp": "2026-11-22T21:17:31.595Z",
                    "ordinal": 3,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "id": "rs_multi_1",
                        "summary": [],
                        "content": None,
                        "encrypted_content": "gAAAAABrMulti1==",
                    },
                },
                {
                    "timestamp": "2026-11-22T21:17:31.596Z",
                    "ordinal": 4,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "id": "rs_multi_2",
                        "summary": [
                            {"type": "summary_text", "text": "First thought part."},
                            "Bare string summary part.",
                        ],
                        "content": None,
                        "encrypted_content": "gAAAAABrMulti2==",
                    },
                },
                {
                    "timestamp": "2026-11-22T21:17:31.598Z",
                    "ordinal": 5,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "id": "rs_multi_3",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "Second consecutive reasoning item.",
                            }
                        ],
                        "content": None,
                        "encrypted_content": "gAAAAABrMulti3==",
                    },
                },
                {
                    "timestamp": "2026-11-22T21:17:31.601Z",
                    "ordinal": 6,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Note: literal <think>tag</think> in prose.",
                            }
                        ],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:17:31.602Z",
                    "ordinal": 7,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "id": "rs_multi_4",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "Bundled reasoning before second message.",
                            }
                        ],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:17:31.605Z",
                    "ordinal": 8,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Dispatching two calls."}],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:17:31.606Z",
                    "ordinal": 9,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "id": "rs_multi_5",
                        "summary": [{"type": "summary_text", "text": "Thought before tool call."}],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:17:31.607Z",
                    "ordinal": 10,
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "status": "completed",
                        "call_id": "call_first",
                        "name": "exec",
                        "input": 'text("first");',
                    },
                },
                {
                    "timestamp": "2026-11-22T21:17:31.608Z",
                    "ordinal": 11,
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "status": "completed",
                        "call_id": "call_second",
                        "name": "exec",
                        "input": 'text("second");',
                    },
                },
                # Return call_second before call_first to verify tool_order sorting.
                {
                    "timestamp": "2026-11-22T21:17:31.630Z",
                    "ordinal": 12,
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "call_id": "call_second",
                        "output": [{"type": "input_text", "text": "second\n"}],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:17:31.640Z",
                    "ordinal": 13,
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "call_id": "call_first",
                        "output": [{"type": "input_text", "text": "first\n"}],
                    },
                },
                {
                    "timestamp": "2026-11-22T21:17:31.643Z",
                    "ordinal": 14,
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_cost": 0.042,
                            "total_token_usage": {
                                "input_tokens": 1200,
                                "cached_input_tokens": 300,
                                "output_tokens": 90,
                                "reasoning_output_tokens": 40,
                                "total_tokens": 1290,
                            },
                            "last_token_usage": {
                                "input_tokens": 1200,
                                "cached_input_tokens": 300,
                                "output_tokens": 90,
                                "reasoning_output_tokens": 40,
                                "total_tokens": 1290,
                            },
                        },
                    },
                },
            ],
        },
    }

    # 4. Standard function_call / function_call_output (gpt-5 tool shape),
    # JSON dict output with metadata, dict without output key, non-dict JSON args,
    # empty dict output, scalar JSON output, plain string output, token_count before
    # output (no-op), cli_version fallback, cost_usd fallback, and final_text fallback.
    function_call_gpt5: dict[str, Any] = {
        "cli_version": "0.152.0",
        "cli_model": "openai/gpt-5.6-sol",
        "input_record": {
            "id": "fc-gpt5",
            "model": "openai/gpt-5.6-sol",
            "cost_usd": 0.015,
            "messages": [
                _session_meta(
                    session_id="sess-fc-1",
                    timestamp="2026-11-22T18:00:00.000Z",
                    originator=None,
                    cli_version=None,
                ),
                # token_count before any output must be ignored by finish_api_call.
                {
                    "timestamp": "2026-11-22T18:00:00.050Z",
                    "ordinal": 1,
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "last_token_usage": {"input_tokens": 1, "output_tokens": 1},
                        },
                    },
                },
                # No task_started before turn_context -> turn_context sets turn_id.
                _turn_context(
                    timestamp="2026-11-22T18:00:00.100Z",
                    ordinal=2,
                    turn_id="turn-fc-ctx",
                    model="gpt-5.6-sol",
                ),
                {
                    "timestamp": "2026-11-22T18:00:00.200Z",
                    "ordinal": 3,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Inspect the directory."}],
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:01.000Z",
                    "ordinal": 4,
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "fc_1",
                        "name": "shell",
                        "arguments": '{"workdir": "/workspace", "command": "ls -la"}',
                        "status": "completed",
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:01.100Z",
                    "ordinal": 5,
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "fc_1",
                        "output": json.dumps(
                            {
                                "output": "total 4\n-rw-r--r-- 1 user user 12 data.txt\n",
                                "metadata": {"exit_code": 0, "duration_seconds": 0.1},
                            }
                        ),
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:01.200Z",
                    "ordinal": 6,
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "fc_2",
                        "name": "verify_factors",
                        "arguments": "[3, 5]",
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:01.250Z",
                    "ordinal": 7,
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "fc_2",
                        "output": '{"correct": true, "score": 1}',
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:01.300Z",
                    "ordinal": 8,
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "fc_3",
                        "name": "noop",
                        "arguments": None,
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:01.350Z",
                    "ordinal": 9,
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "fc_3",
                        "output": "{}",
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:01.400Z",
                    "ordinal": 10,
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "fc_4",
                        "name": "plain_tool",
                        "arguments": '{"k": "v"}',
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:01.450Z",
                    "ordinal": 11,
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "fc_4",
                        "output": "plain stdout",
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:01.500Z",
                    "ordinal": 12,
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "fc_5",
                        "name": "scalar_tool",
                        "arguments": '{"n": 1}',
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:01.550Z",
                    "ordinal": 13,
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "fc_5",
                        "output": "42",
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:01.600Z",
                    "ordinal": 14,
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 400,
                                "cached_input_tokens": 50,
                                "output_tokens": 60,
                                "reasoning_output_tokens": 10,
                                "total_tokens": 460,
                            },
                            "last_token_usage": {
                                "input_tokens": 400,
                                "cached_input_tokens": 50,
                                "output_tokens": 60,
                                "reasoning_output_tokens": 10,
                                "total_tokens": 460,
                            },
                        },
                    },
                },
                {
                    "timestamp": "2026-11-22T18:00:02.000Z",
                    "ordinal": 15,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "<answer>15</answer>"}],
                    },
                },
            ],
        },
    }

    # 5. web_search_call + turn_aborted + task_complete with error.
    web_search_and_abort: dict[str, Any] = {
        "cli_version": "0.154.0",
        "cli_model": "openai/gpt-5.6-sol",
        "input_record": {
            "id": "web-abort",
            "model": "openai/gpt-5.6-sol",
            "messages": [
                _session_meta(
                    session_id="sess-web-1",
                    timestamp="2026-11-22T19:00:00.000Z",
                ),
                {
                    "timestamp": "2026-11-22T19:00:00.010Z",
                    "ordinal": 1,
                    "type": "event_msg",
                    "payload": {"type": "turn_started", "turn_id": "turn-web-1"},
                },
                _turn_context(
                    timestamp="2026-11-22T19:00:00.020Z",
                    ordinal=2,
                    turn_id="turn-web-1",
                    model="gpt-5.6-sol",
                ),
                {
                    "timestamp": "2026-11-22T19:00:00.030Z",
                    "ordinal": 3,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Search for prime records."}],
                    },
                },
                {
                    "timestamp": "2026-11-22T19:00:01.000Z",
                    "ordinal": 4,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "Searching the web."}],
                    },
                },
                {
                    "timestamp": "2026-11-22T19:00:01.100Z",
                    "ordinal": 5,
                    "type": "response_item",
                    "payload": {
                        "type": "web_search_call",
                        "status": "completed",
                        "action": {
                            "type": "search",
                            "query": "largest known prime",
                            "queries": ["largest known prime", "mersenne prime"],
                            "url": "https://example.invalid/primes",
                        },
                    },
                },
                {
                    "timestamp": "2026-11-22T19:00:01.200Z",
                    "ordinal": 6,
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 300,
                                "output_tokens": 40,
                                "total_tokens": 340,
                            },
                            "last_token_usage": {
                                "input_tokens": 300,
                                "output_tokens": 40,
                                "total_tokens": 340,
                            },
                        },
                    },
                },
                {
                    "timestamp": "2026-11-22T19:00:01.300Z",
                    "ordinal": 7,
                    "type": "event_msg",
                    "payload": {"type": "turn_aborted", "reason": "interrupted"},
                },
                {
                    "timestamp": "2026-11-22T19:00:02.000Z",
                    "ordinal": 8,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Partial answer before error."}
                        ],
                    },
                },
                {
                    "timestamp": "2026-11-22T19:00:02.100Z",
                    "ordinal": 9,
                    "type": "event_msg",
                    "payload": {
                        "type": "task_complete",
                        "turn_id": "turn-web-1",
                        "last_agent_message": None,
                        "error": {
                            "message": '{"error":{"message":"Budget has been exceeded!","type":"budget_exceeded","param":null,"code":"400"}}'
                        },
                    },
                },
            ],
        },
    }

    # 6. Orphan outputs and a trailing unanswered call (killed mid-tool),
    # plus token_count events with missing info / last_token_usage.
    orphan_and_unanswered: dict[str, Any] = {
        "cli_version": "0.154.0",
        "cli_model": "openai/gpt-5.6-luna",
        "input_record": {
            "id": "orphan-killed",
            "model": "openai/gpt-5.6-luna",
            "session_id": "sess-orphan-1",
            "messages": [
                _session_meta(
                    session_id="sess-orphan-1",
                    timestamp="2026-11-22T20:00:00.000Z",
                ),
                _turn_context(
                    timestamp="2026-11-22T20:00:00.100Z",
                    ordinal=1,
                    turn_id="turn-orphan-1",
                    model="gpt-5.6-luna",
                ),
                {
                    "timestamp": "2026-11-22T20:00:00.200Z",
                    "ordinal": 2,
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "call_from_earlier_tail",
                        "output": "orphan result text",
                    },
                },
                {
                    "timestamp": "2026-11-22T20:00:00.250Z",
                    "ordinal": 3,
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "call_id": None,
                        "output": None,
                    },
                },
                {
                    "timestamp": "2026-11-22T20:00:01.000Z",
                    "ordinal": 4,
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "call_id": "call_answered",
                        "name": "exec",
                        "input": 'text("ok");',
                    },
                },
                {
                    "timestamp": "2026-11-22T20:00:01.100Z",
                    "ordinal": 5,
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "call_id": "call_answered",
                        "output": [{"type": "input_text", "text": "ok\n"}],
                    },
                },
                {
                    "timestamp": "2026-11-22T20:00:01.200Z",
                    "ordinal": 6,
                    "type": "event_msg",
                    "payload": {"type": "token_count", "info": None},
                },
                {
                    "timestamp": "2026-11-22T20:00:02.000Z",
                    "ordinal": 7,
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "Starting slow command."}],
                    },
                },
                {
                    "timestamp": "2026-11-22T20:00:02.050Z",
                    "ordinal": 8,
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "call_id": "call_killed_mid_tool",
                        "name": "exec",
                        "input": 'await tools.exec_command({cmd:"sleep 9999"});',
                    },
                },
                {
                    "timestamp": "2026-11-22T20:00:02.100Z",
                    "ordinal": 9,
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 250,
                                "output_tokens": 30,
                            },
                            "last_token_usage": None,
                        },
                    },
                },
            ],
        },
    }

    # 7. Empty rollout -> placeholder step, version "unknown" fallback.
    empty_rollout: dict[str, Any] = {
        "cli_version": "",
        "cli_model": "openai/gpt-5.6-luna",
        "input_record": {},
    }

    # 8. Malformed / unknown events, non-ISO timestamps, missing call_id,
    # non-list message content, bare string content, non-standard role, non-list
    # reasoning summary, non-string tool arguments, and base64 blob rendering.
    b64_blob = "QWxhZGRpbjpvcGVuIHNlc2FtZQ" * 12
    malformed_and_unknown: dict[str, Any] = {
        "cli_version": "0.154.0",
        "cli_model": "openai/gpt-5.6-luna",
        "input_record": {
            "id": "malformed",
            "model": "openai/gpt-5.6-luna",
            "session_id": "sess-fallback-from-record",
            "messages": [
                "not-a-dict",
                123,
                None,
                {"type": "unknown_event_type", "payload": {"x": 1}},
                {"type": "event_msg", "payload": "not-a-dict"},
                {"type": "event_msg", "payload": {"type": "unknown_event_msg"}},
                {"type": "response_item", "payload": "not-a-dict"},
                {"type": "response_item", "payload": {"type": "unknown_response_item"}},
                {
                    "timestamp": "not-an-iso-timestamp",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "tool",
                        "content": ["bare-text ", {"type": "input_text", "text": "and dict"}],
                    },
                },
                {
                    "timestamp": 1234567890,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": "non-list-content-becomes-empty",
                    },
                },
                {
                    "timestamp": "2026-11-22T21:00:00.000Z",
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "summary": "not-a-list",
                    },
                },
                {
                    "timestamp": "2026-11-22T21:00:00.100Z",
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "summary": [{"text": 999}, None],
                    },
                },
                # Missing call_id is skipped.
                {
                    "timestamp": "2026-11-22T21:00:00.200Z",
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": "",
                        "name": "ignored_no_id",
                        "arguments": "{}",
                    },
                },
                # Non-string non-None arguments hits raw_args else branch -> {"value": 123}.
                {
                    "timestamp": "2026-11-22T21:00:00.300Z",
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "call_id": "call_non_str_args",
                        "name": "weird_tool",
                        "input": 123,
                    },
                },
                {
                    "timestamp": "2026-11-22T21:00:00.400Z",
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call_output",
                        "call_id": "call_non_str_args",
                        "output": [{"text": "part1"}, "raw-part2"],
                    },
                },
                {
                    "timestamp": "bad-ts-on-assistant",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": f"Payload:\n{b64_blob}"}],
                    },
                },
            ],
        },
    }

    return {
        "multi_turn_exec_reasoning": multi_turn_exec,
        "encrypted_only_reasoning": encrypted_only,
        "consecutive_reasoning_and_bundled": consecutive_reasoning,
        "function_call_gpt5": function_call_gpt5,
        "web_search_and_abort": web_search_and_abort,
        "orphan_and_unanswered": orphan_and_unanswered,
        "empty_rollout": empty_rollout,
        "malformed_and_unknown": malformed_and_unknown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Codex trajectory fixtures from original CheatBench."
    )
    parser.add_argument(
        "--cheatbench",
        type=Path,
        default=DEFAULT_CHEATBENCH,
        help=f"Path to a read-only checkout of CheatBench at {UPSTREAM_COMMIT[:7]}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for JSON fixture files.",
    )
    args = parser.parse_args()

    cases = _build_cases()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    with tempfile.TemporaryDirectory(prefix="cb-codex-fixtures-") as tmpdir:
        proc = subprocess.run(
            [sys.executable, "-B", "-c", _EVAL_SCRIPT, str(args.cheatbench.resolve())],
            input=json.dumps(cases),
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    generated: dict[str, dict[str, Any]] = json.loads(proc.stdout)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, fixture in generated.items():
        dest = args.output_dir / f"{name}.json"
        dest.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(generated)} fixtures to {args.output_dir}")


if __name__ == "__main__":
    main()
