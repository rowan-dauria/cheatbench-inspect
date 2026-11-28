# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/tasks/go/environment/server/service.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

"""Isolated `engine` container entry point for the Go game and KataGo service."""

# Runs inside the Go engine container's /opt/engine/venv.
# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import hmac
import importlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import redirect_stderr, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_KATAGO = "/opt/engine/katago/squashfs-root/AppRun"
DEFAULT_NETWORK = "/opt/engine/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
DOUBLE_KATAGO = "/opt/engine/katago-double"
DOUBLE_NETWORK = "/opt/engine/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
MAX_REQUEST_BYTES = 1048576


def query_gpu_info(*, required: bool) -> dict[str, Any]:
    """Check NVIDIA GPU availability and return GPU name, driver, and memory."""
    error = (
        "The default Go build requires an NVIDIA GPU for its CUDA KataGo backend. "
        "No NVIDIA GPU was detected inside the engine container. "
        "Run on a GPU-capable host or Hawk cluster, or use the CPU test-double engine image "
        "for local tests."
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        if required:
            raise RuntimeError(error)
        return {"available": False, "reason": "CUDA_VISIBLE_DEVICES disabled"}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if required:
            raise RuntimeError(error) from exc
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        if required:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"{error} (nvidia-smi rc={result.returncode}: {detail})")
        return {"available": False, "reason": result.stderr.strip() or f"rc={result.returncode}"}
    parts = [part.strip() for part in lines[0].split(",")]
    return {
        "available": True,
        "name": parts[0] if len(parts) > 0 else "unknown",
        "driver_version": parts[1] if len(parts) > 1 else "unknown",
        "memory_total": parts[2] if len(parts) > 2 else "unknown",
        "raw": lines,
    }


def _resolve_engine_paths(
    katago_path: str | None, network_path: str | None
) -> tuple[str, str, bool]:
    """Return (binary, network, is_test_double)."""
    binary = katago_path or os.environ.get("KATAGO_BINARY")
    network = network_path or os.environ.get("KATAGO_NETWORK")
    is_double = os.environ.get("GO_ENGINE_TEST_DOUBLE") == "1"
    if not binary:
        binary = DOUBLE_KATAGO if is_double and Path(DOUBLE_KATAGO).is_file() else DEFAULT_KATAGO
    if not network:
        network = (
            DOUBLE_NETWORK if is_double and Path(DOUBLE_NETWORK).is_file() else DEFAULT_NETWORK
        )
    return binary, network, is_double


class EngineService:
    def __init__(
        self,
        directory: Path,
        fixture: dict[str, Any],
        output: Path,
        *,
        timeout_s: int = 3600,
        katago_path: str | None = None,
        network_path: str | None = None,
        skip_gpu_check: bool = False,
    ) -> None:
        self.directory = directory
        self.fixture = fixture
        self.output = output
        self.timeout_s = timeout_s
        self.binary, self.network, self.is_double = _resolve_engine_paths(katago_path, network_path)
        self.gpu_info = query_gpu_info(required=not (skip_gpu_check or self.is_double))
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.runner: Any = None
        self.engine: Any = None
        self.signature: dict[str, Any] | None = None
        self.sequence = 0
        self.fatal_error: str | None = None
        self.lock = threading.Lock()
        try:
            self.wait(self._start())
        except BaseException:
            if self.engine is not None:
                with suppress(Exception):
                    self.wait(self.engine.close())
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            raise

    def wait(self, coroutine: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout=900)
        except TimeoutError:
            future.cancel()
            raise

    async def _start(self) -> None:
        katago_mod = importlib.import_module("game.katago_api")
        sig_mod = importlib.import_module("game.engine_signature")
        ep_mod = importlib.import_module("game.episode")

        self.engine = katago_mod.KataGoEngine(
            self.binary,
            self.network,
            max_visits=400,
            work_dir=self.output,
        )
        await self.engine.start()
        self.signature = await sig_mod.verify(
            self.engine,
            signature_path=self.directory / "engine_signature.json",
            probes_dir=self.directory / "probes",
        )
        self.runner = ep_mod.Episode(
            self.fixture,
            self.output,
            self.engine,
            timeout=self.timeout_s,
            prompt_path=self.directory / "instruction.txt",
        )
        self.runner.engine_signature = self.signature
        (self.output / "engine_signature.json").write_text(
            json.dumps(self.signature, indent=2), encoding="utf-8"
        )
        (self.output / "prompt.md").write_text(self.runner.build_prompt() + "\n", encoding="utf-8")

    async def _dispatch(
        self, name: str, arguments: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        runner = self.runner
        if self.sequence == 0:
            runner.started = time.monotonic()
            runner.deadline = runner.started + min(3600, max(1, float(timeout_s)))
        self.sequence += 1
        call_id = str(self.sequence)
        runner.active_tool_call_id = call_id
        runner.event("shared_tool_requested", name=name, arguments=arguments, call_id=call_id)
        try:
            result = await runner.dispatch(name, arguments)
            runner.message({"role": "tool", "tool_call_id": call_id, "content": result})
            return {
                "content": [{"type": "text", "text": result}],
                "isError": False,
                "game_state": runner.game.result(),
            }
        finally:
            (self.output / "game.json").write_text(
                json.dumps(runner.game.result(), indent=2), encoding="utf-8"
            )
            (self.output / "accounting.json").write_text(
                json.dumps(self._accounting(), indent=2), encoding="utf-8"
            )

    def _accounting(self) -> dict[str, int]:
        runner = self.runner
        if runner is None:
            return {
                "api_requests": 0,
                "endpoint_requests": 0,
                "health_requests": 0,
                "tool_calls": 0,
            }
        return {
            "api_requests": len(runner.api_requests),
            "endpoint_requests": len(runner.engine_api_requests),
            "health_requests": sum(r["service"] == "health" for r in runner.api_requests),
            "tool_calls": self.sequence,
        }

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        name, arguments = payload.get("name"), payload.get("arguments")
        names = {s["function"]["name"] for s in self.runner.build_tools()}
        if not isinstance(name, str) or name not in names or not isinstance(arguments, dict):
            raise ValueError("Invalid tool call")
        with self.lock:
            if self.fatal_error:
                raise RuntimeError(self.fatal_error)
            return self.wait(
                self._dispatch(name, arguments, payload.get("timeout_s", self.timeout_s))
            )

    def snapshot(self) -> dict[str, Any]:
        runner = self.runner
        surface_summary = runner.surfaces.finalize() if runner is not None else {}
        final_snapshot = (
            {path: file.text for path, file in runner.surfaces._files.items()}
            if runner is not None
            else {}
        )
        diff_path = self.output / "surface_diff.json"
        surface_diff = (
            json.loads(diff_path.read_text(encoding="utf-8")) if diff_path.is_file() else {}
        )
        accounting = self._accounting()
        return {
            "engine_signature": self.signature,
            "gpu_info": self.gpu_info,
            "is_test_double": self.is_double,
            "game_result": runner.game.result() if runner is not None else None,
            "accounting": accounting,
            "endpoint_requests": accounting["endpoint_requests"],
            "health_requests": accounting["health_requests"],
            "api_requests": [
                {
                    "service": r.get("service"),
                    "game": r.get("game"),
                    "arguments": r.get("arguments"),
                    "status_code": (r.get("result") or {}).get("status_code"),
                }
                for r in (runner.api_requests if runner is not None else [])
            ],
            "surface_summary": surface_summary,
            "final_snapshot": final_snapshot,
            "surface_diff": surface_diff,
            "infrastructure_error": self.fatal_error,
        }

    def close(self) -> None:
        if self.engine is not None:
            with suppress(Exception):
                self.wait(self.engine.close())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


def make_handler(service: EngineService, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "engine/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _reject(self) -> None:
            body = json.dumps({"error": "Not found."}).encode("utf-8")
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/harness/call" or not hmac.compare_digest(
                self.headers.get("X-Harness-Token", ""), token
            ):
                self._reject()
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size")
                answer = service.call(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError):
                answer = {
                    "content": [{"type": "text", "text": "Invalid Go tool request."}],
                    "isError": True,
                }
            except Exception as exc:
                service.fatal_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"Go infrastructure error: {service.fatal_error}", file=sys.stderr, flush=True
                )
                (service.output / "infrastructure_error.json").write_text(
                    json.dumps({"error": service.fatal_error}), encoding="utf-8"
                )
                answer = {"infrastructure_error": "Go game/engine service failed; episode invalid."}
            data = json.dumps(answer).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._reject()

        def do_PUT(self) -> None:
            self._reject()

        def do_DELETE(self) -> None:
            self._reject()

        def do_PATCH(self) -> None:
            self._reject()

        def do_HEAD(self) -> None:
            self._reject()

        def do_OPTIONS(self) -> None:
            self._reject()

    return Handler


def serve(
    directory: Path,
    fixture: dict[str, Any],
    token: str,
    *,
    port: int = 8765,
    timeout_s: int = 3600,
    katago_path: str | None = None,
    network_path: str | None = None,
    skip_gpu_check: bool = False,
) -> None:
    os.umask(0o077)
    sys.path.insert(0, str(directory))
    out_dir = directory / "state"
    out_dir.mkdir(parents=True, exist_ok=True)
    svc = EngineService(
        directory,
        fixture,
        out_dir,
        timeout_s=timeout_s,
        katago_path=katago_path,
        network_path=network_path,
        skip_gpu_check=skip_gpu_check,
    )
    stop_watch = threading.Event()
    try:
        ThreadingHTTPServer.allow_reuse_address = True
        httpd = ThreadingHTTPServer(
            ("0.0.0.0", port),
            make_handler(svc, token),
            bind_and_activate=False,
        )
        httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        httpd.server_bind()
        httpd.server_activate()
        bound_port = httpd.server_address[1]

        def watch() -> None:
            while not stop_watch.wait(1.0):
                proc = svc.engine._proc if svc.engine is not None else None
                if proc is not None and proc.returncode is not None:
                    svc.fatal_error = f"KataGo exited {proc.returncode}"
                    (svc.output / "infrastructure_error.json").write_text(
                        json.dumps({"error": svc.fatal_error}), encoding="utf-8"
                    )
                    httpd.shutdown()
                    return
                if svc.fatal_error:
                    httpd.shutdown()
                    return

        with (
            httpd,
            (directory / "engine.log").open("w", encoding="utf-8") as diagnostics,
            redirect_stderr(diagnostics),
        ):
            server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            server_thread.start()
            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
            try:
                print(f"http://engine:{bound_port}", flush=True)
                sys.stdin.read()
            finally:
                stop_watch.set()
                httpd.shutdown()
                server_thread.join(timeout=10)
    finally:
        captured = svc.snapshot()
        svc.close()

    print(json.dumps(captured), flush=True)
