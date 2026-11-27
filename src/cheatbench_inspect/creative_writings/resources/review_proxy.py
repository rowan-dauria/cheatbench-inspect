# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/review_server.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

"""Loopback HTTP adapter for the original grader, run as root in the sandbox."""

import json
import secrets
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from typing import Any


def serve(token: str) -> None:
    # Inspect installs the RPC client in a root-only directory. The HTTP shim
    # submits only the original review fields and returns the public reply.
    sys.path.append("/var/tmp/sandbox-services/editorial")
    call: Callable[..., dict[str, Any]] = import_module("editorial").call_editorial

    class ReviewHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "editorial/1.0"
        sys_version = ""
        timeout = 15

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def send_json(self, body: dict[str, Any], status: int) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/review":
                self.send_json({"error": "not found"}, 404)
                return
            header = self.headers.get("Authorization", "")
            if not secrets.compare_digest(header.encode(), ("Bearer " + token).encode()):
                self.send_json({"error": "unauthorised"}, 401)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("missing body")
                request = json.loads(self.rfile.read(length).decode("utf-8", "replace"))
                if not isinstance(request, dict):
                    raise ValueError("expected object")
            except (OSError, ValueError):
                self.send_json({"error": "bad request"}, 400)
                return

            try:
                reply = call(
                    "review",
                    brief=str(request.get("task_brief") or ""),
                    reference=str(request.get("reference_text", request.get("page_text")) or ""),
                    submission=str(request.get("submission_text") or ""),
                )
            except Exception:
                reply = {"verdict": None, "feedback": "review temporarily unavailable; try again"}
            self.send_json(reply, 503 if reply["verdict"] is None else 200)

    with ThreadingHTTPServer(("127.0.0.1", 0), ReviewHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"http://127.0.0.1:{server.server_port}/review", flush=True)
        # The solver owns the subprocess and closes it at the episode boundary.
        sys.stdin.read()
        server.shutdown()
