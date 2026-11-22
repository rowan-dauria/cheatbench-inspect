"""Unit and loopback HTTP tests for the in-sandbox Codex Responses proxy."""

from __future__ import annotations

import ast
import base64
import http.client
import http.server
import json
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any, cast

from inspect_ai.agent._bridge.responses_impl import messages_from_responses_input
from inspect_ai.model import ContentText
from inspect_ai.model._internal import CONTENT_INTERNAL_TAG, content_internal_tag
from inspect_ai.model._openai_responses import openai_responses_inputs

from cheatbench_inspect.reasoning_replay import RESPONSES_REASONING_MARKER
from cheatbench_inspect.resources.codex import proxy


def _make_think_item(
    *,
    item_id: str,
    summary: str | None,
    encrypted_content: str | None,
    readable: str | None = None,
) -> dict[str, Any]:
    marker_b64 = base64.b64encode(
        json.dumps(
            {
                proxy.RESPONSES_REASONING_MARKER: {
                    "id": item_id,
                    "summary": summary,
                    "encrypted_content": encrypted_content,
                    "content": readable,
                }
            }
        ).encode("utf-8")
    ).decode("ascii")
    summary_xml = f"<summary>{summary}</summary>\n" if summary else ""
    body = encrypted_content or readable or ""
    return {
        "id": "msg_bridge_uuid",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": (
                    f'<think signature="{item_id}" redacted="true" internal="{marker_b64}">\n'
                    f"{summary_xml}{body}\n</think>"
                ),
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


def _parse_sse(chunks: list[bytes]) -> list[tuple[str, dict[str, Any]]]:
    raw = b"".join(chunks).decode("utf-8")
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.strip().split("\n\n"):
        etype = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                etype = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        assert etype is not None
        assert data is not None
        events.append((etype, data))
    return events


def test_proxy_is_stdlib_only_and_neutral() -> None:
    """Verify proxy.py imports only stdlib modules and contains no eval-identifying strings."""
    source = Path(proxy.__file__).read_text(encoding="utf-8")
    lower = source.lower()
    for term in ("cheatbench", "inspect", "eval"):
        assert term not in lower
    assert proxy.RESPONSES_REASONING_MARKER == RESPONSES_REASONING_MARKER

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top in sys.stdlib_module_names, f"non-stdlib import: {top}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            assert top in sys.stdlib_module_names, f"non-stdlib import: {top}"


def test_outbound_reasoning_and_tool_calls_preserved() -> None:
    resp = {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": [
            _make_think_item(item_id="rs_1", summary=None, encrypted_content="gAAAA_1"),
            _make_think_item(
                item_id="rs_2",
                summary="Tricky </summary> and <think>literal</think> summary",
                encrypted_content="gAAAA_2",
            ),
            {
                "id": "msg_prose",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Literal <think>not a reasoning block</think> in prose.",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
            {
                "id": "ctc_1",
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "exec",
                "namespace": "functions",
                "input": "tools.exec_command({cmd:'pwd'})",
                "status": "completed",
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_2",
                "name": "wait",
                "namespace": "functions",
                "arguments": '{"cell_id":"c1"}',
                "status": "completed",
            },
        ],
    }

    rewritten = proxy.rewrite_response_payload(resp)
    items = rewritten["output"]
    assert len(items) == 5

    assert items[0] == {
        "id": "rs_1",
        "type": "reasoning",
        "summary": [],
        "status": "completed",
        "encrypted_content": "gAAAA_1",
        "content": None,
    }
    assert items[1] == {
        "id": "rs_2",
        "type": "reasoning",
        "summary": [
            {
                "type": "summary_text",
                "text": "Tricky </summary> and <think>literal</think> summary",
            }
        ],
        "status": "completed",
        "encrypted_content": "gAAAA_2",
        "content": None,
    }
    # Prose with literal <think> and no <content-internal> is untouched.
    assert items[2] == resp["output"][2]
    assert items[3] == resp["output"][3]
    assert items[4] == resp["output"][4]


def test_content_internal_outbound_and_inbound_variants() -> None:
    tag_both = content_internal_tag({"message_id": "msg_both", "message_phase": "commentary"})
    tag_id_only = content_internal_tag({"message_id": "msg_id_only"})
    tag_phase_only = content_internal_tag({"message_phase": "final_answer"})

    resp = {
        "id": "resp_ci",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "rand1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"Running command.\n{tag_both}\n",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
            {
                "id": "rand2",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"No phase here.\n{tag_id_only}\n",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
            {
                "id": "rand3",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"Final answer.\n{tag_phase_only}\n",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
            {
                "id": "rand4",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Plain message without metadata.",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
            {
                "id": "rand5",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"Literal <{CONTENT_INTERNAL_TAG}>not-valid!</{CONTENT_INTERNAL_TAG}> tag.",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
            {
                "id": "rand6",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "refusal",
                        "refusal": f"Cannot comply.\n{tag_both}\n",
                    }
                ],
            },
        ],
    }

    rewritten = proxy.rewrite_response_payload(resp)
    out = rewritten["output"]

    assert out[0]["id"] == "msg_both"
    assert out[0]["phase"] == "commentary"
    assert out[0]["content"][0]["text"] == "Running command."

    assert out[1]["id"] == "msg_id_only"
    assert "phase" not in out[1]
    assert out[1]["content"][0]["text"] == "No phase here."

    assert out[2]["id"] == "rand3"
    assert out[2]["phase"] == "final_answer"
    assert out[2]["content"][0]["text"] == "Final answer."

    # Message without content-internal left completely unchanged.
    assert out[3] == resp["output"][3]
    # Malformed tag left untouched as literal text.
    assert out[4] == resp["output"][4]

    # Refusal part stripped and id/phase lifted.
    assert out[5]["id"] == "msg_both"
    assert out[5]["phase"] == "commentary"
    assert out[5]["content"][0]["refusal"] == "Cannot comply."


async def test_content_internal_inbound_restores_bridge_internal() -> None:
    """Rebuilt inbound tags parse back to identical ContentText.internal and wire fields."""
    req = {
        "model": "openai/gpt-5.6-luna",
        "store": False,
        "input": [
            {
                "type": "message",
                "id": "msg_user_1",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
            {
                "type": "message",
                "id": "msg_asst_1",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Working on it."}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "No id or phase."}],
            },
        ],
    }

    rewritten = proxy.rewrite_request_payload(req)
    # User message untouched even though it has an id.
    assert rewritten["input"][0] == req["input"][0]
    # Assistant message with no id/phase untouched.
    assert rewritten["input"][2] == req["input"][2]

    # Assistant message with id+phase carries exact bridge tag format and round-trips through Inspect.
    asst_text = rewritten["input"][1]["content"][0]["text"]
    assert asst_text.startswith("Working on it.\n<content-internal>")
    assert asst_text.endswith("</content-internal>\n")

    # Idempotent: calling rewrite_request_payload a second time does not double-tag.
    assert proxy.rewrite_request_payload(rewritten) == rewritten

    messages = messages_from_responses_input(rewritten["input"], [], "openai/gpt-5.6-luna")
    asst_parts = [c for c in messages[1].content if isinstance(c, ContentText)]
    assert asst_parts[0].text == "Working on it."
    assert asst_parts[0].internal == {
        "message_id": "msg_asst_1",
        "message_phase": "commentary",
    }
    assert asst_parts[1].text == "No id or phase."
    assert asst_parts[1].internal is None

    wire = await openai_responses_inputs(messages)
    asst_wire = [
        cast(dict[str, Any], it)
        for it in wire
        if isinstance(it, dict) and it.get("role") == "assistant"
    ]
    assert asst_wire[0]["id"] == "msg_asst_1"
    assert asst_wire[0]["phase"] == "commentary"
    assert "id" not in asst_wire[1]
    assert "phase" not in asst_wire[1]


def test_internal_store_fallback_for_unprefixed_id_and_extra_keys() -> None:
    """When Codex strips a non-prefixed id, the per-episode store recovers it."""
    store = proxy._InternalStore()
    tag_unprefixed = content_internal_tag(
        {"message_id": "bareuuid1234", "message_phase": "commentary"}
    )
    tag_extra = content_internal_tag(
        {"message_id": "msg_extra_1", "message_phase": "final_answer", "custom": "v"}
    )
    out = proxy.rewrite_response_payload(
        {
            "output": [
                {
                    "id": "rand1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": f"Step one.\n{tag_unprefixed}\n"}],
                },
                {
                    "id": "rand2",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": f"Step two.\n{tag_extra}\n"}],
                },
            ]
        },
        internal_store=store,
    )
    assert out["output"][0]["id"] == "bareuuid1234"
    assert out["output"][0]["phase"] == "commentary"

    # Codex strips non-prefixed id on item 0, keeps prefixed id on item 1.
    inbound = proxy.rewrite_request_payload(
        {
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Step one."}],
                },
                {
                    "type": "message",
                    "id": "msg_extra_1",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "Step two."}],
                },
            ]
        },
        internal_store=store,
    )
    msgs = messages_from_responses_input(inbound["input"], [], "openai/gpt-5.6-luna")
    parts = [c for c in msgs[0].content if isinstance(c, ContentText)]
    assert parts[0].internal == {"message_id": "bareuuid1234", "message_phase": "commentary"}
    assert parts[1].internal == {
        "message_id": "msg_extra_1",
        "message_phase": "final_answer",
        "custom": "v",
    }


def test_iter_responses_sse_events_and_phase() -> None:
    resp = {
        "id": "resp_sse",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "rs_sse_1",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Thinking briefly."}],
                "encrypted_content": "gAAAA_SSE",
                "content": None,
                "status": "completed",
            },
            {
                "id": "msg_sse_1",
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Calling tools.",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
            {
                "id": "ctc_sse_1",
                "type": "custom_tool_call",
                "call_id": "call_sse_1",
                "name": "exec",
                "namespace": "functions",
                "input": "echo 1",
                "status": "completed",
            },
            {
                "id": "fc_sse_1",
                "type": "function_call",
                "call_id": "call_sse_2",
                "name": "wait",
                "arguments": '{"ms":10}',
                "status": "completed",
            },
        ],
    }

    events = _parse_sse(list(proxy.iter_responses_sse(resp)))
    types = [etype for etype, _ in events]
    assert types[0] == "response.created"
    assert types[1] == "response.in_progress"
    assert types[-1] == "response.completed"
    assert events[0][1]["response"]["output"] == []
    assert [ev["sequence_number"] for _, ev in events] == list(range(1, len(events) + 1))

    assert "response.reasoning_summary_part.added" in types
    assert "response.reasoning_summary_text.delta" in types
    assert "response.reasoning_summary_text.done" in types
    assert "response.reasoning_summary_part.done" in types
    assert "response.custom_tool_call_input.done" in types
    assert "response.function_call_arguments.done" in types

    msg_done = next(
        ev["item"]
        for etype, ev in events
        if etype == "response.output_item.done" and ev["item"]["type"] == "message"
    )
    assert msg_done["id"] == "msg_sse_1"
    assert msg_done["phase"] == "commentary"


def test_loopback_http_proxy_with_fake_bridge() -> None:
    """Exercise streaming, non-streaming, errors, passthrough paths, and safe logging."""
    seen_requests: list[tuple[str, str, Any]] = []
    secret_sentinel = "SUPER_SECRET_BODY_SENTINEL_999"

    tag_commentary = content_internal_tag(
        {"message_id": "msg_http_1", "message_phase": "commentary"}
    )

    class _FakeBridgeHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            del format, args

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            seen_requests.append(("GET", self.path, None))
            self._send_json(200, {"data": [{"id": "gpt-5.6-luna"}]})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8"))
            seen_requests.append(("POST", self.path, body))

            if self.path == "/v1/responses/compact":
                self._send_json(200, {"compacted": True, "echo": body})
                return
            if body.get("model") == "trigger-400":
                self._send_json(
                    400, {"error": {"message": "bad request", "type": "invalid_request_error"}}
                )
                return
            if body.get("model") == "trigger-500":
                self._send_json(500, {"error": {"message": "internal failure"}})
                return

            self._send_json(
                200,
                {
                    "id": "resp_http_1",
                    "object": "response",
                    "status": "completed",
                    "model": body.get("model"),
                    "output": [
                        _make_think_item(
                            item_id="rs_http_1",
                            summary="HTTP test summary",
                            encrypted_content="gAAAA_HTTP_ENC",
                        ),
                        {
                            "id": "bridge_random_uuid",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": f"Hello {secret_sentinel}\n{tag_commentary}\n",
                                    "annotations": [],
                                    "logprobs": [],
                                }
                            ],
                        },
                    ],
                },
            )

    class _FakeBridge(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    bridge = _FakeBridge(("127.0.0.1", 0), _FakeBridgeHandler)
    bridge_thread = threading.Thread(target=bridge.serve_forever, daemon=True)
    bridge_thread.start()

    log_lines: list[str] = []
    srv = proxy.ResponsesProxy(
        f"http://127.0.0.1:{bridge.server_address[1]}/v1",
        log=log_lines.append,
    ).start()
    try:
        # 1. Non-streaming POST /v1/responses: outbound reasoning + content-internal normalized.
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
        conn.request(
            "POST",
            "/v1/responses",
            body=json.dumps({"model": "test", "input": [], "stream": False}),
            headers={"Content-Type": "application/json"},
        )
        r = conn.getresponse()
        assert r.status == 200
        doc = json.loads(r.read().decode("utf-8"))
        conn.close()
        assert doc["output"][0]["type"] == "reasoning"
        assert doc["output"][0]["id"] == "rs_http_1"
        assert doc["output"][0]["summary"] == [
            {"type": "summary_text", "text": "HTTP test summary"}
        ]
        assert doc["output"][0]["encrypted_content"] == "gAAAA_HTTP_ENC"
        assert doc["output"][1]["type"] == "message"
        assert doc["output"][1]["id"] == "msg_http_1"
        assert doc["output"][1]["phase"] == "commentary"
        assert doc["output"][1]["content"][0]["text"] == f"Hello {secret_sentinel}"

        # 2. Streaming POST /v1/responses with Turn-2 assistant message carrying id/phase:
        #    proxy sends stream=False and restored <content-internal> to bridge, returns SSE.
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
        conn.request(
            "POST",
            "/v1/responses",
            body=json.dumps(
                {
                    "model": "test",
                    "stream": True,
                    "input": [
                        {
                            "type": "message",
                            "id": "msg_http_1",
                            "role": "assistant",
                            "phase": "commentary",
                            "content": [
                                {"type": "output_text", "text": f"Hello {secret_sentinel}"}
                            ],
                        }
                    ],
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        r = conn.getresponse()
        assert r.status == 200
        assert "text/event-stream" in (r.getheader("Content-Type") or "")
        sse_raw = r.read()
        conn.close()
        assert seen_requests[-1][2]["stream"] is False
        forwarded_text = seen_requests[-1][2]["input"][0]["content"][0]["text"]
        assert forwarded_text == f"Hello {secret_sentinel}\n{tag_commentary}\n"
        events = _parse_sse([sse_raw])
        assert events[-1][0] == "response.completed"
        assert events[-1][1]["response"]["output"][0]["type"] == "reasoning"
        assert events[-1][1]["response"]["output"][1]["phase"] == "commentary"

        # 3. 4xx and 5xx passthrough unchanged.
        for code, model_name in ((400, "trigger-400"), (500, "trigger-500")):
            conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
            conn.request(
                "POST",
                "/v1/responses",
                body=json.dumps({"model": model_name, "input": [], "stream": True}),
                headers={"Content-Type": "application/json"},
            )
            r = conn.getresponse()
            assert r.status == code
            err_doc = json.loads(r.read().decode("utf-8"))
            conn.close()
            assert "error" in err_doc

        # 4. Other paths (/v1/responses/compact, /v1/models) pass through unchanged.
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
        compact_body = {"input": [{"role": "user", "content": secret_sentinel}]}
        conn.request(
            "POST",
            "/v1/responses/compact",
            body=json.dumps(compact_body),
            headers={"Content-Type": "application/json"},
        )
        r = conn.getresponse()
        assert r.status == 200
        assert json.loads(r.read().decode("utf-8")) == {"compacted": True, "echo": compact_body}
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
        conn.request("GET", "/v1/models")
        r = conn.getresponse()
        assert r.status == 200
        assert json.loads(r.read().decode("utf-8")) == {"data": [{"id": "gpt-5.6-luna"}]}
        conn.close()

        # 5. Logs contain method, path, and status ONLY — never bodies.
        assert log_lines == [
            "POST /v1/responses -> 200",
            "POST /v1/responses -> 200",
            "POST /v1/responses -> 400",
            "POST /v1/responses -> 500",
            "POST /v1/responses/compact -> 200",
            "GET /v1/models -> 200",
        ]
        assert all(secret_sentinel not in line for line in log_lines)
    finally:
        srv.close()
        bridge.shutdown()
        bridge.server_close()


def test_start_helper_contract() -> None:
    """Verify proxy.start(base_url, config) satisfies episode.py's interface."""
    srv = proxy.start("http://127.0.0.1:19999/v1", {"timeout": 30})
    try:
        assert srv.url.startswith("http://127.0.0.1:")
        assert srv.url.endswith("/v1")
    finally:
        srv.close()
