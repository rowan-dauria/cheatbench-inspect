# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/runner_support.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

"""Episode-local MCP grader transport from CheatBench's core/agents/runner_support.py."""

from __future__ import annotations

import hmac
import json
import secrets
import sys
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast


def _log(message: str) -> None:
    print(f"[runner] {message}", file=sys.stderr, flush=True)


MCP_PROTOCOL_VERSION = "2025-06-18"


class _GraderMCPHandler(BaseHTTPRequestHandler):
    """Streamable-HTTP MCP, the four methods a client actually calls."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        """Silence. The default writes to stderr, which the host parses."""

    def _authorised(self) -> bool:
        path = self.path.split("?", 1)[0]
        return hmac.compare_digest(path, cast("GraderServer", self.server).grader_path)

    def _empty(self, code: int) -> None:
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, body: bytes) -> None:
        if "text/event-stream" in (self.headers.get("Accept") or ""):
            payload = b"event: message\ndata: " + body + b"\n\n"
            ctype = "text/event-stream"
        else:
            (payload, ctype) = (body, "application/json")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._empty(404 if not self._authorised() else 405)

    def do_DELETE(self) -> None:
        self._empty(404 if not self._authorised() else 405)

    def do_POST(self) -> None:
        if not self._authorised():
            self._empty(404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._empty(400)
            return
        raw = self.rfile.read(length) if length else b""
        try:
            request = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self._empty(400)
            return
        if not isinstance(request, dict):
            self._empty(400)
            return
        if request.get("id") is None:
            self._empty(202)
            return
        self._send(json.dumps(self._handle(request)).encode())

    def _handle(self, request: dict) -> dict:
        (rid, method) = (request.get("id"), request.get("method"))
        params = request.get("params")
        params = params if isinstance(params, dict) else {}
        server = cast("GraderServer", self.server)

        def ok(result):
            return {"jsonrpc": "2.0", "id": rid, "result": result}

        if method == "initialize":
            server.connected = True
            return ok(
                {
                    "protocolVersion": params.get("protocolVersion") or MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": server.grader_server_name, "version": "1.0.0"},
                }
            )
        if method == "ping":
            return ok({})
        if method == "tools/list":
            server.listed = True
            return ok(
                {
                    "tools": [
                        {k: v for k, v in spec.items() if k != "call"}
                        for spec in server.grader_tools.values()
                    ]
                }
            )
        if method == "tools/call":
            name = params.get("name")
            spec = server.grader_tools.get(name) if isinstance(name, str) else None
            if spec is None:
                return ok(
                    {
                        "content": [{"type": "text", "text": f"no such tool: {name}"}],
                        "isError": True,
                    }
                )
            args = params.get("arguments")
            try:
                return ok(spec["call"](args if isinstance(args, dict) else {}))
            except Exception as e:
                _log(f"grader raised: {type(e).__name__}: {e}")
                server.grader_errors.append(f"{type(e).__name__}: {e}")
                return ok(
                    {
                        "content": [{"type": "text", "text": "the grader is unavailable"}],
                        "isError": True,
                    }
                )
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32601, "message": f"unknown method {method}"},
        }


class GraderServer(ThreadingHTTPServer):
    """The running grader endpoint, and the handle the episode holds it by."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        *,
        state: dict,
        call: Callable[[dict], dict],
        server_name: str,
        tool_name: str,
        description: str,
        schema: dict,
        extra_tools: Sequence[dict] = (),
    ):
        tools = [
            {"name": tool_name, "description": description, "inputSchema": schema, "call": call},
            *extra_tools,
        ]
        names = [spec["name"] for spec in tools]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Grader tool names must be nonempty strings")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate grader tool names")
        if any(not callable(spec["call"]) for spec in tools):
            raise ValueError("Each grader tool needs a callable handler")
        super().__init__(("127.0.0.1", 0), _GraderMCPHandler)
        self.grader_state = state
        self.grader_call = call
        self.grader_server_name = server_name
        self.grader_tool_name = tool_name
        self.grader_description = description
        self.grader_schema = schema
        self.grader_tools = {spec["name"]: spec for spec in tools}
        self.grader_errors: list[str] = []
        self.grader_path = "/" + secrets.token_hex(16) + "/mcp"
        self.connected = False
        self.listed = False
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        (host, port) = (self.server_address[0], self.server_address[1])
        return f"http://{host}:{port}{self.grader_path}"

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Stop serving. Idempotent; see the Codex runner for why it must be."""
        if self._thread is not None:
            with suppress(Exception):
                self.shutdown()
            self._thread.join(timeout=10)
            self._thread = None
        with suppress(Exception):
            self.server_close()

    def report(self) -> dict:
        entry = {
            "name": self.grader_server_name,
            "transport": "streamable-http",
            "status": "connected" if self.connected else "configured",
            "tools_listed": self.listed,
        }
        if self.grader_errors:
            entry["errors"] = list(self.grader_errors)
        return entry
