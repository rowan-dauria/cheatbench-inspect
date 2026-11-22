"""In-sandbox loopback proxy for OpenAI Responses reasoning and message metadata.

Stdlib only. Sits between the agent CLI and the local model bridge:

  CLI -> http://127.0.0.1:<proxy_port>/v1/responses
      -> http://127.0.0.1:<bridge_port>/v1/responses (stream=False)

Outbound (bridge to CLI):
- Rewrites assistant ``message`` items carrying a marked ``<think>`` tag into
  native Responses ``type="reasoning"`` output items at the same index.
- Strips bridge ``<content-internal>`` tags from assistant ``message`` text and
  moves ``message_id`` and ``message_phase`` (when present) onto the item's
  native ``id`` and ``phase`` fields.
- Synthesizes Responses SSE events when the client requested ``stream=True``.

Inbound (CLI to bridge):
- Rebuilds ``<content-internal>`` tags on assistant ``message`` input items from
  their native ``id`` and ``phase`` fields (with a per-episode fallback map when
  an upstream ID is non-prefixed and stripped by the CLI, or carries extra
  metadata keys), appending the tag in the bridge's exact format.

All other paths (such as ``/v1/responses/compact`` or ``/v1/models``) and all
non-2xx responses pass through unchanged. Only method, path and status are
logged; bodies are never logged.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import html
import http.client
import http.server
import json
import re
import socketserver
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlsplit

RESPONSES_REASONING_MARKER = "responses_reasoning_v1"
CONTENT_INTERNAL_TAG = "content-internal"
_MAX_INTERNAL_DEPTH = 100

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

_THINK_RE = re.compile(r"^\s*<think([^>]*)>.*</think>\s*$", re.DOTALL)
_INTERNAL_ATTR_RE = re.compile(r'internal="([^"]*)"')
_CONTENT_INTERNAL_RE = re.compile(
    rf"\n?<{CONTENT_INTERNAL_TAG}>([A-Za-z0-9+/=\s]*)</{CONTENT_INTERNAL_TAG}>\n?"
)
_CONTENT_INTERNAL_BARE_RE = re.compile(
    rf"<{CONTENT_INTERNAL_TAG}>([A-Za-z0-9+/=\s]*)</{CONTENT_INTERNAL_TAG}>"
)


def _default_log(msg: str) -> None:
    print(f"[responses-proxy] {msg}", file=sys.stderr, flush=True)


def _exceeds_max_depth(value: Any, max_depth: int, depth: int = 0) -> bool:
    if depth > max_depth:
        return True
    if isinstance(value, dict):
        return any(_exceeds_max_depth(v, max_depth, depth + 1) for v in value.values())
    if isinstance(value, list):
        return any(_exceeds_max_depth(v, max_depth, depth + 1) for v in value)
    return False


def _decode_b64_json(encoded: str) -> tuple[bool, Any]:
    """Decode base64-encoded UTF-8 JSON within the depth cap."""
    try:
        decoded = json.loads(
            base64.b64decode("".join(encoded.split()), validate=True).decode("utf-8")
        )
    except (ValueError, TypeError, binascii.Error, UnicodeDecodeError):
        return False, None
    if _exceeds_max_depth(decoded, _MAX_INTERNAL_DEPTH):
        return False, None
    return True, decoded


def _encode_content_internal_tag(internal: Any) -> str:
    encoded = base64.b64encode(json.dumps(internal).encode("utf-8")).decode("utf-8")
    return f"<{CONTENT_INTERNAL_TAG}>{encoded}</{CONTENT_INTERNAL_TAG}>"


def _has_valid_content_internal(text: str) -> bool:
    for match in _CONTENT_INTERNAL_BARE_RE.finditer(text):
        ok, _ = _decode_b64_json(match.group(1))
        if ok:
            return True
    return False


def _strip_content_internal(text: str) -> tuple[str, bool, Any]:
    """Remove valid ``<content-internal>`` tags from text.

    Returns ``(cleaned_text, found, first_decoded)``. Invalid tags are left
    untouched as literal text.
    """
    decoded_values: list[Any] = []

    def _replace(match: re.Match[str]) -> str:
        ok, value = _decode_b64_json(match.group(1))
        if not ok:
            return match.group(0)
        decoded_values.append(value)
        return ""

    stripped = _CONTENT_INTERNAL_RE.sub(_replace, text)
    if not decoded_values:
        return text, False, None
    return stripped, True, decoded_values[0]


def _is_prefixed_id(item_id: str) -> bool:
    prefix, sep, suffix = item_id.partition("_")
    return bool(sep and prefix and suffix)


def _decode_reasoning_marker(text: str) -> dict[str, Any] | None:
    """Return the packed reasoning payload if ``text`` is a marked ``<think>`` tag."""
    match = _THINK_RE.match(text)
    if not match:
        return None
    attr_match = _INTERNAL_ATTR_RE.search(match.group(1))
    if not attr_match:
        return None
    ok, decoded = _decode_b64_json(html.unescape(attr_match.group(1)))
    if not ok or not isinstance(decoded, dict):
        return None
    payload = decoded.get(RESPONSES_REASONING_MARKER)
    return payload if isinstance(payload, dict) else None


class _InternalStore:
    """Per-episode fallback store for assistant message metadata.

    The CLI echoes both ``id`` and ``phase`` on assistant ``message`` items when
    ``id`` is underscore-prefixed (for example ``msg_...``). This store only
    records entries that cannot be reconstructed from native ``id`` and
    ``phase`` alone: non-prefixed ``message_id`` values (which the CLI drops
    before sending a request) or internal dicts with keys beyond ``message_id``
    and ``message_phase``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_text_phase: dict[tuple[str, str | None], dict[str, Any]] = {}

    def record(
        self,
        *,
        clean_text: str,
        item_id: str | None,
        phase: str | None,
        internal: dict[str, Any],
    ) -> None:
        extra_keys = set(internal.keys()) - {"message_id", "message_phase"}
        msg_id = internal.get("message_id")
        unprefixed_id = isinstance(msg_id, str) and bool(msg_id) and not _is_prefixed_id(msg_id)
        if not extra_keys and not unprefixed_id:
            return
        with self._lock:
            if extra_keys and item_id:
                self._by_id[item_id] = dict(internal)
            if unprefixed_id or extra_keys:
                self._by_text_phase[(clean_text, phase)] = dict(internal)

    def lookup(
        self,
        *,
        text: str,
        item_id: str | None,
        phase: str | None,
    ) -> dict[str, Any] | None:
        with self._lock:
            if item_id and item_id in self._by_id:
                return dict(self._by_id[item_id])
            if not item_id:
                stored = self._by_text_phase.get((text, phase))
                if stored is not None:
                    return dict(stored)
        return None


def _rewrite_message_content_internal(
    item: dict[str, Any],
    internal_store: _InternalStore | None = None,
) -> dict[str, Any]:
    """Strip ``<content-internal>`` tags from an assistant message and lift id/phase."""
    content = item.get("content")
    if not isinstance(content, list):
        return item

    new_content: list[Any] = []
    changed = False
    first_internal: Any = None
    first_found = False
    first_clean_text: str | None = None

    for part in content:
        if not isinstance(part, dict):
            new_content.append(part)
            continue
        part_type = part.get("type")
        field = (
            "text" if part_type == "output_text" else "refusal" if part_type == "refusal" else None
        )
        updated = part
        if field is not None and isinstance(part.get(field), str):
            cleaned, found, decoded = _strip_content_internal(part[field])
            if found:
                changed = True
                updated = {**part, field: cleaned}
                if not first_found:
                    first_found = True
                    first_internal = decoded
                    first_clean_text = cleaned
        new_content.append(updated)

    if not changed:
        return item

    rewritten = {**item, "content": new_content}
    if isinstance(first_internal, dict):
        msg_id = first_internal.get("message_id")
        if isinstance(msg_id, str) and msg_id:
            rewritten["id"] = msg_id
        msg_phase = first_internal.get("message_phase")
        if isinstance(msg_phase, str) and msg_phase:
            rewritten["phase"] = msg_phase
        if internal_store is not None and first_clean_text is not None:
            internal_store.record(
                clean_text=first_clean_text,
                item_id=rewritten.get("id") if isinstance(rewritten.get("id"), str) else None,
                phase=rewritten.get("phase") if isinstance(rewritten.get("phase"), str) else None,
                internal=first_internal,
            )
    return rewritten


def rewrite_output_item(
    item: dict[str, Any],
    internal_store: _InternalStore | None = None,
) -> dict[str, Any]:
    """Rewrite a bridge assistant message into native reasoning or clean message."""
    if item.get("type") != "message" or item.get("role") != "assistant":
        return item
    content = item.get("content")
    if isinstance(content, list) and len(content) == 1:
        part = content[0]
        if isinstance(part, dict) and part.get("type") == "output_text":
            text = part.get("text")
            if isinstance(text, str):
                payload = _decode_reasoning_marker(text)
                if payload is not None:
                    summary_text = payload.get("summary")
                    summary_list: list[dict[str, str]] = (
                        [{"type": "summary_text", "text": summary_text}]
                        if isinstance(summary_text, str) and summary_text
                        else []
                    )
                    native: dict[str, Any] = {
                        "id": payload.get("id") or item.get("id") or "rs_proxy",
                        "type": "reasoning",
                        "summary": summary_list,
                        "status": "completed",
                    }
                    enc = payload.get("encrypted_content")
                    if isinstance(enc, str) and enc:
                        native["encrypted_content"] = enc
                    raw_content = payload.get("content")
                    if isinstance(raw_content, str) and raw_content:
                        native["content"] = [{"type": "reasoning_text", "text": raw_content}]
                    else:
                        native["content"] = None
                    return native
    return _rewrite_message_content_internal(item, internal_store=internal_store)


def rewrite_response_payload(
    resp: dict[str, Any],
    internal_store: _InternalStore | None = None,
) -> dict[str, Any]:
    """Return a shallow copy of ``resp`` with reasoning and metadata normalized."""
    output = resp.get("output")
    if not isinstance(output, list):
        return resp
    new_output = [
        rewrite_output_item(item, internal_store=internal_store) if isinstance(item, dict) else item
        for item in output
    ]
    return {**resp, "output": new_output}


def _resolve_inbound_internal(
    item: dict[str, Any],
    part_text: str,
    internal_store: _InternalStore | None,
) -> dict[str, Any] | None:
    raw_id = item.get("id")
    msg_id = raw_id if isinstance(raw_id, str) and raw_id else None
    raw_phase = item.get("phase")
    msg_phase = raw_phase if isinstance(raw_phase, str) and raw_phase else None

    if internal_store is not None:
        stored = internal_store.lookup(text=part_text, item_id=msg_id, phase=msg_phase)
        if stored is not None:
            merged = dict(stored)
            if msg_id is not None:
                merged["message_id"] = msg_id
            if msg_phase is not None:
                merged["message_phase"] = msg_phase
            return merged

    internal: dict[str, Any] = {}
    if msg_id is not None:
        internal["message_id"] = msg_id
    if msg_phase is not None:
        internal["message_phase"] = msg_phase
    return internal or None


def rewrite_input_item(
    item: dict[str, Any],
    internal_store: _InternalStore | None = None,
) -> dict[str, Any]:
    """Rebuild ``<content-internal>`` tags on an inbound assistant message."""
    if item.get("role") != "assistant" or item.get("type", "message") != "message":
        return item

    content = item.get("content")
    if isinstance(content, str):
        if _has_valid_content_internal(content):
            return item
        internal = _resolve_inbound_internal(item, content, internal_store)
        if not internal:
            return item
        tag = _encode_content_internal_tag(internal)
        return {**item, "content": f"{content}\n{tag}\n"}

    if not isinstance(content, list):
        return item

    new_content: list[Any] = []
    changed = False
    for part in content:
        if not isinstance(part, dict):
            new_content.append(part)
            continue
        part_type = part.get("type")
        field = (
            "text"
            if part_type in ("output_text", "input_text")
            else "refusal"
            if part_type == "refusal"
            else None
        )
        updated = part
        if field is not None and isinstance(part.get(field), str):
            text = part[field]
            if not _has_valid_content_internal(text):
                internal = _resolve_inbound_internal(item, text, internal_store)
                if internal:
                    tag = _encode_content_internal_tag(internal)
                    updated = {**part, field: f"{text}\n{tag}\n"}
                    changed = True
        new_content.append(updated)

    return {**item, "content": new_content} if changed else item


def rewrite_request_payload(
    req: dict[str, Any],
    internal_store: _InternalStore | None = None,
) -> dict[str, Any]:
    """Return a shallow copy of ``req`` with assistant ``<content-internal>`` restored."""
    items = req.get("input")
    if not isinstance(items, list):
        return req
    changed = False
    new_items: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            rewritten = rewrite_input_item(item, internal_store=internal_store)
            if rewritten is not item:
                changed = True
            new_items.append(rewritten)
        else:
            new_items.append(item)
    return {**req, "input": new_items} if changed else req


def _sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _iter_chunks(text: str, max_len: int = 256) -> Iterator[str]:
    for i in range(0, len(text), max_len):
        yield text[i : i + max_len]


def iter_responses_sse(resp: dict[str, Any]) -> Iterator[bytes]:
    """Synthesize OpenAI Responses SSE events from a complete Response dict."""
    seq_num = 0

    def emit(event_type: str, body: dict[str, Any]) -> bytes:
        nonlocal seq_num
        seq_num += 1
        payload = {**body, "type": event_type, "sequence_number": seq_num}
        return _sse_event(event_type, payload)

    in_progress_resp = {
        **resp,
        "status": "in_progress",
        "output": [],
        "usage": None,
        "incomplete_details": None,
    }
    yield emit("response.created", {"response": in_progress_resp})
    yield emit("response.in_progress", {"response": in_progress_resp})

    for output_index, output_item in enumerate(resp.get("output", [])):
        if not isinstance(output_item, dict):
            continue
        item_id = output_item.get("id", f"item_{output_index}")
        item_type = output_item.get("type")

        added_item = dict(output_item)
        if "status" in added_item:
            added_item["status"] = "in_progress"
        if item_type == "message":
            added_item["content"] = []
        elif item_type == "reasoning":
            added_item["summary"] = []
            if added_item.get("content"):
                added_item["content"] = []
        elif item_type == "custom_tool_call":
            added_item["input"] = ""
        elif item_type == "function_call":
            added_item["arguments"] = ""

        yield emit(
            "response.output_item.added",
            {"item": added_item, "output_index": output_index},
        )

        if item_type == "reasoning":
            for r_idx, r_part in enumerate(output_item.get("content") or []):
                if isinstance(r_part, dict) and r_part.get("type") == "reasoning_text":
                    text = r_part.get("text", "")
                    for chunk in _iter_chunks(text):
                        yield emit(
                            "response.reasoning_text.delta",
                            {
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": r_idx,
                                "delta": chunk,
                            },
                        )
                    yield emit(
                        "response.reasoning_text.done",
                        {
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": r_idx,
                            "text": text,
                        },
                    )
            for s_idx, s_part in enumerate(output_item.get("summary") or []):
                if not isinstance(s_part, dict):
                    continue
                empty_part = dict(s_part)
                if empty_part.get("type") == "summary_text":
                    empty_part["text"] = ""
                yield emit(
                    "response.reasoning_summary_part.added",
                    {
                        "item_id": item_id,
                        "output_index": output_index,
                        "summary_index": s_idx,
                        "part": empty_part,
                    },
                )
                if s_part.get("type") == "summary_text":
                    text = s_part.get("text", "")
                    for chunk in _iter_chunks(text):
                        yield emit(
                            "response.reasoning_summary_text.delta",
                            {
                                "item_id": item_id,
                                "output_index": output_index,
                                "summary_index": s_idx,
                                "delta": chunk,
                            },
                        )
                    yield emit(
                        "response.reasoning_summary_text.done",
                        {
                            "item_id": item_id,
                            "output_index": output_index,
                            "summary_index": s_idx,
                            "text": text,
                        },
                    )
                yield emit(
                    "response.reasoning_summary_part.done",
                    {
                        "item_id": item_id,
                        "output_index": output_index,
                        "summary_index": s_idx,
                        "part": s_part,
                    },
                )

        elif item_type == "message":
            for c_idx, content in enumerate(output_item.get("content") or []):
                if not isinstance(content, dict):
                    continue
                ctype = content.get("type")
                empty_part = dict(content)
                if ctype == "output_text":
                    empty_part["text"] = ""
                elif ctype == "refusal":
                    empty_part["refusal"] = ""
                yield emit(
                    "response.content_part.added",
                    {
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": c_idx,
                        "part": empty_part,
                    },
                )
                if ctype == "output_text":
                    text = content.get("text", "")
                    for chunk in _iter_chunks(text):
                        yield emit(
                            "response.output_text.delta",
                            {
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": c_idx,
                                "delta": chunk,
                                "logprobs": [],
                            },
                        )
                    yield emit(
                        "response.output_text.done",
                        {
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": c_idx,
                            "text": text,
                            "logprobs": [],
                        },
                    )
                elif ctype == "refusal":
                    refusal = content.get("refusal", "")
                    for chunk in _iter_chunks(refusal):
                        yield emit(
                            "response.refusal.delta",
                            {
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": c_idx,
                                "delta": chunk,
                            },
                        )
                    yield emit(
                        "response.refusal.done",
                        {
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": c_idx,
                            "refusal": refusal,
                        },
                    )
                yield emit(
                    "response.content_part.done",
                    {
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": c_idx,
                        "part": content,
                    },
                )

        elif item_type == "custom_tool_call":
            input_data = output_item.get("input", "")
            for chunk in _iter_chunks(input_data):
                yield emit(
                    "response.custom_tool_call_input.delta",
                    {
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": chunk,
                    },
                )
            yield emit(
                "response.custom_tool_call_input.done",
                {
                    "item_id": item_id,
                    "output_index": output_index,
                    "input": input_data,
                },
            )

        elif item_type == "function_call":
            arguments = output_item.get("arguments", "")
            for chunk in _iter_chunks(arguments):
                yield emit(
                    "response.function_call_arguments.delta",
                    {
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": chunk,
                    },
                )
            yield emit(
                "response.function_call_arguments.done",
                {
                    "item_id": item_id,
                    "output_index": output_index,
                    "arguments": arguments,
                },
            )

        elif item_type == "web_search_call":
            for suffix in ("in_progress", "searching", "completed"):
                yield emit(
                    f"response.web_search_call.{suffix}",
                    {"item_id": item_id, "output_index": output_index},
                )

        done_item = dict(output_item)
        if "status" in done_item or item_type == "custom_tool_call":
            done_item["status"] = "completed"
        yield emit(
            "response.output_item.done",
            {"item": done_item, "output_index": output_index},
        )

    completed_resp = {**resp, "status": "completed"}
    yield emit("response.completed", {"response": completed_resp})


class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: ResponsesProxy  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _read_body(self) -> bytes | None:
        length = self.headers.get("Content-Length")
        if length is not None:
            n = int(length)
            return self.rfile.read(n) if n else b""
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            out = bytearray()
            while True:
                line = self.rfile.readline().strip()
                size = int(line.split(b";", 1)[0], 16) if line else 0
                if size == 0:
                    while True:
                        trailer = self.rfile.readline()
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    return bytes(out)
                out += self.rfile.read(size)
                self.rfile.readline()
        return None

    def _forward_request(
        self, method: str, path: str, headers: Any, body: bytes | None
    ) -> tuple[int, str, list[tuple[str, str]], bytes]:
        out_headers: dict[str, str] = {}
        for name, value in headers.items():
            if name.lower() in _HOP_BY_HOP:
                continue
            out_headers[name] = value
        out_headers["Host"] = self.server.upstream_netloc
        if body is not None:
            out_headers["Content-Length"] = str(len(body))
        elif method.upper() in ("POST", "PUT", "PATCH"):
            out_headers["Content-Length"] = "0"

        conn = http.client.HTTPConnection(
            self.server.upstream_host,
            self.server.upstream_port,
            timeout=self.server.timeout,
        )
        try:
            conn.request(method, path, body=body, headers=out_headers)
            resp = conn.getresponse()
            raw = resp.read()
            return resp.status, resp.reason, list(resp.getheaders()), raw
        finally:
            conn.close()

    def _handle(self) -> None:
        body = self._read_body()
        req_path = urlsplit(self.path).path.rstrip("/") or "/"
        is_responses_post = self.command == "POST" and req_path.endswith("/responses")
        client_wants_stream = False
        forward_body = body

        if is_responses_post and body:
            try:
                req_doc = json.loads(body.decode("utf-8"))
                if isinstance(req_doc, dict):
                    client_wants_stream = bool(req_doc.get("stream", False))
                    rewritten_req = rewrite_request_payload(
                        req_doc, internal_store=self.server.internal_store
                    )
                    if client_wants_stream:
                        rewritten_req = {**rewritten_req, "stream": False}
                    if rewritten_req is not req_doc:
                        forward_body = json.dumps(rewritten_req, ensure_ascii=False).encode("utf-8")
            except (ValueError, UnicodeDecodeError):
                is_responses_post = False

        try:
            status, reason, resp_headers, raw = self._forward_request(
                self.command, self.path, self.headers, forward_body
            )
        except Exception as exc:
            err = json.dumps(
                {
                    "error": {
                        "type": "proxy_error",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                }
            ).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            self.wfile.flush()
            self.server.log(f"{self.command} {req_path} -> 502")
            return

        if is_responses_post and 200 <= status < 300:
            try:
                resp_doc = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                resp_doc = None
            if isinstance(resp_doc, dict) and isinstance(resp_doc.get("output"), list):
                rewritten = rewrite_response_payload(
                    resp_doc, internal_store=self.server.internal_store
                )
                if client_wants_stream:
                    self.send_response(200, "OK")
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    try:
                        for event_bytes in iter_responses_sse(rewritten):
                            self.wfile.write(b"%x\r\n" % len(event_bytes) + event_bytes + b"\r\n")
                            self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        self.close_connection = True
                    self.server.log(f"{self.command} {req_path} -> 200")
                    return

                out_bytes = json.dumps(rewritten, ensure_ascii=False).encode("utf-8")
                self.send_response(status, reason)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out_bytes)))
                self.end_headers()
                self.wfile.write(out_bytes)
                self.wfile.flush()
                self.server.log(f"{self.command} {req_path} -> {status}")
                return

        self.send_response(status, reason)
        for name, value in resp_headers:
            if name.lower() not in _HOP_BY_HOP:
                self.send_header(name, value)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()
        self.server.log(f"{self.command} {req_path} -> {status}")

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = _handle  # noqa: N815


class ResponsesProxy(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Loopback proxy normalizing Responses reasoning items and message metadata."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        upstream_base_url: str,
        *,
        host: str = "127.0.0.1",
        timeout: float | None = 3600.0,
        log: Callable[[str], None] | None = None,
    ) -> None:
        parts = urlsplit(upstream_base_url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(f"Expected http://host:port[/path], got {upstream_base_url!r}")
        self.upstream_host = parts.hostname
        self.upstream_port = parts.port or 80
        self.upstream_netloc = parts.netloc
        self.upstream_path = parts.path.rstrip("/")
        self.timeout = timeout
        self.log = log or _default_log
        self.internal_store = _InternalStore()
        super().__init__((host, 0), _ProxyHandler)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.server_address[0]!s}:{self.port}{self.upstream_path}"

    def start(self) -> ResponsesProxy:
        self._thread = threading.Thread(
            target=self.serve_forever, name="responses-proxy", daemon=True
        )
        self._thread.start()
        return self

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown()
        self.server_close()


def start(base_url: str, config: dict[str, Any] | None = None) -> ResponsesProxy:
    """Start the Responses loopback proxy for the runner."""
    cfg = config or {}
    timeout_raw = cfg.get("timeout", 3600.0)
    timeout = float(timeout_raw) if timeout_raw is not None else None
    host = str(cfg.get("host") or "127.0.0.1")
    return ResponsesProxy(base_url, host=host, timeout=timeout).start()
