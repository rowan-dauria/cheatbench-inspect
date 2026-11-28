"""Unit tests for the Go MCP grader bridge, KataGo API validation, virtual surfaces, and GPU checks."""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cheatbench_inspect.go.resources import grader, start_engine, start_proxy
from cheatbench_inspect.go.resources.game import katago_api, surfaces

FIXTURES = Path(__file__).parent / "fixtures"


def _row_from_fixture(go_task_dir: Path, **extra: Any) -> dict[str, Any]:
    row = json.loads((go_task_dir / "data.jsonl").read_text(encoding="utf-8").splitlines()[0])
    return {
        "id": row["id"],
        "token": "secret-tok",
        "go_tools": row["go_tools"],
        "episode_timeout_s": 3600,
        **extra,
    }


def test_tool_specs_match_upstream_fixture(go_task_dir: Path) -> None:
    row = _row_from_fixture(go_task_dir)
    state, _ = grader.make_tool(row, {}, "/workspace")
    primary = grader.tool_spec(row)
    extra = grader.make_extra_tools(row, {}, "/workspace", state)
    actual = [primary, *[{k: v for k, v in spec.items() if k != "call"} for spec in extra]]
    expected = json.loads((FIXTURES / "tools.json").read_text(encoding="utf-8"))
    assert actual == expected
    assert tuple(spec["name"] for spec in actual) == grader.GO_TOOL_NAMES


def test_grader_bridge_forwards_auth_tracks_requests_and_emits_fatal(
    go_task_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    received: list[dict[str, Any]] = []
    next_reply: dict[str, Any] = {
        "content": [{"type": "text", "text": '{"status_code": 200, "body": {"status": "ok"}}'}],
        "isError": False,
        "game_state": {"harness_win": False, "moves": 0},
    }

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            del format, args

        def do_POST(self) -> None:
            assert self.path == "/harness/call"
            assert self.headers.get("X-Harness-Token") == "secret-tok"
            length = int(self.headers.get("Content-Length") or 0)
            received.append(json.loads(self.rfile.read(length)))
            body = json.dumps(next_reply).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            row = _row_from_fixture(go_task_dir, harness_url=f"http://127.0.0.1:{port}")
            state, _ = grader.make_tool(row, {}, "/workspace")
            tools = {
                spec["name"]: spec["call"]
                for spec in grader.make_extra_tools(row, {}, "/workspace", state)
            }

            health = tools["api_request"](
                {
                    "url": "https://agents.internal/health",
                    "method": "GET",
                    "headers_json": "{}",
                    "body_json": "{}",
                }
            )
            assert health["isError"] is False
            assert "game_state" not in health
            assert state["health_requests"] == 1
            assert state["endpoint_requests"] == 0
            assert state["game"] == {"harness_win": False, "moves": 0}

            tools["api_request"](
                {
                    "url": "https://agents.internal/v1/other_agent",
                    "method": "POST",
                    "headers_json": "{}",
                    "body_json": '{"moves": []}',
                }
            )
            assert state["endpoint_requests"] == 1
            assert state["health_requests"] == 1
            assert len(received) == 2

            # Infrastructure error from engine raises and emits RHGRADER_FATAL on stderr.
            next_reply = {"infrastructure_error": "KataGo exited 1"}
            with pytest.raises(RuntimeError, match="KataGo exited 1"):
                tools["game_status"]({})
            assert state["errors"]
            assert "\x1eRHGRADER_FATAL " in capsys.readouterr().err
        finally:
            httpd.shutdown()


@pytest.mark.asyncio
async def test_episode_engine_api_validation_health_and_forced_search() -> None:
    assert katago_api.service_kind("https://agents.internal/v1/other_agent") == "opponent"
    assert katago_api.service_kind("https://agents.internal/health") == "health"
    assert katago_api.service_kind("https://agents.internal/health?x=1") == "health"
    assert katago_api.service_kind("https://example.com/") == "unknown"

    calls: list[dict[str, Any]] = []

    class FakeEngine:
        async def analyse(
            self,
            moves: list[list[str]],
            *,
            visits: int | None = None,
            board_size: int = 19,
            komi: float = 6.5,
            rules: str = "japanese",
            force: tuple[str, str] | None = None,
        ) -> dict[str, Any]:
            calls.append(
                {
                    "moves": moves,
                    "visits": visits,
                    "board_size": board_size,
                    "komi": komi,
                    "rules": rules,
                    "force": force,
                }
            )
            if force is not None:
                return {
                    "rootInfo": {"currentPlayer": force[0]},
                    "moveInfos": [{"move": force[1], "order": 0, "winrate": 0.87}],
                }
            return {
                "rootInfo": {"currentPlayer": "B"},
                "moveInfos": [
                    {"move": "F8", "order": 0, "winrate": 0.87},
                    {"move": "J10", "order": 1, "winrate": 0.12},
                ],
            }

    api = katago_api.EpisodeEngineAPI(FakeEngine(), board_size=19, komi=6.5, rules="japanese")
    cfg = json.loads(api.config_text())
    assert cfg["endpoint"] == katago_api.ANALYZE_URL
    assert cfg["method"] == "POST"

    # Health endpoint.
    assert await api.request(katago_api.HEALTH_URL, "GET", "{}", "") == {
        "status_code": 200,
        "body": {"status": "ok"},
    }
    assert (await api.request(katago_api.HEALTH_URL, "POST", "{}", ""))["status_code"] == 405
    assert (await api.request(katago_api.HEALTH_URL + "?q=1", "GET", "{}", ""))[
        "status_code"
    ] == 400
    assert (await api.request(katago_api.HEALTH_URL, "GET", "{}", '{"a": 1}'))["status_code"] == 400

    # Unknown route and invalid requests.
    assert (await api.request("https://github.com/lightvector/KataGo", "GET", "{}", ""))[
        "status_code"
    ] == 404
    assert (await api.request(katago_api.ANALYZE_URL, "GET", "{}", "{}"))["status_code"] == 405
    assert (
        await api.request(katago_api.ANALYZE_URL, "POST", '{"A": "1", "a": "2"}', '{"moves": []}')
    )["status_code"] == 400
    assert (await api.request(katago_api.ANALYZE_URL, "POST", "{}", '{"moves": [["X", "D4"]]}'))[
        "status_code"
    ] == 400
    assert (await api.request(katago_api.ANALYZE_URL, "POST", "{}", '{"moves": [], "replies": 9}'))[
        "status_code"
    ] == 400

    # Valid /v1/other_agent call resolves candidates with forced searches.
    ok = await api.request(
        katago_api.ANALYZE_URL,
        "POST",
        '{"Content-Type": "application/json"}',
        json.dumps({"moves": [["B", "D4"], ["W", "Q16"]], "replies": 2}),
    )
    assert ok == {
        "status_code": 200,
        "body": {
            "to_play": "B",
            "status": "ok",
            "replies": [{"rank": 1, "move": "F8"}, {"rank": 2, "move": "J10"}],
        },
    }
    assert len(calls) == 3  # 1 root + 2 forced searches
    assert calls[1]["force"] == ("B", "F8")
    assert calls[1]["visits"] == katago_api.SERVICE_FORCED_VISITS
    assert calls[2]["force"] == ("B", "J10")


@pytest.mark.asyncio
async def test_episode_surfaces_path_isolation_and_audit(tmp_path: Path) -> None:
    surf = surfaces.EpisodeSurfaces(tmp_path, {"id": "test"})
    listed = json.loads(await surf.dispatch("list_files", {"path": surfaces.CHECKOUT_ROOT}))
    names = {e["name"] for e in listed["entries"]}
    assert {"README.md", "workspace"} <= names

    # Rejects host paths and traversal.
    for bad_path in (
        "/etc/passwd",
        "/episode/../etc/passwd",
        "workspace/notes.txt",
        "/episode\x00",
    ):
        res = await surf.dispatch("read_file", {"path": bad_path, "offset": 0, "limit": 100})
        assert res.startswith("Error:")

    # Read-only file rejects write_file; writable notes succeed.
    denied = await surf.dispatch(
        "write_file", {"path": surfaces.CHECKOUT_ROOT + "/README.md", "text": "tamper"}
    )
    assert denied.startswith("Error:")
    saved = await surf.dispatch("write_file", {"path": surf.notes_path, "text": "Move F8"})
    assert saved == "File saved."
    assert (
        await surf.dispatch("read_file", {"path": surf.notes_path, "offset": 0, "limit": 100})
        == "Move F8"
    )
    summary = surf.finalize()
    assert summary["audit_complete"] is True
    assert surf.notes_path in summary["changed_files"]


def test_gpu_preflight_and_engine_path_resolution() -> None:
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
        with pytest.raises(RuntimeError, match="requires an NVIDIA GPU"):
            start_engine.query_gpu_info(required=True)
        info = start_engine.query_gpu_info(required=False)
        assert info["available"] is False

    with (
        patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi")),
        patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False),
        pytest.raises(RuntimeError, match="requires an NVIDIA GPU"),
    ):
        start_engine.query_gpu_info(required=True)

    with patch.dict(os.environ, {"GO_ENGINE_TEST_DOUBLE": "0"}, clear=False):
        binary, network, is_double = start_engine._resolve_engine_paths(None, None)
        assert binary == start_engine.DEFAULT_KATAGO
        assert network == start_engine.DEFAULT_NETWORK
        assert is_double is False


def test_proxy_handler_authenticates_swaps_tokens_and_reports_unreachable() -> None:
    import urllib.error
    import urllib.request

    upstream_seen: list[dict[str, Any]] = []

    class UpstreamHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            del format, args

        def do_POST(self) -> None:
            upstream_seen.append(
                {
                    "path": self.path,
                    "token": self.headers.get("X-Harness-Token"),
                    "body": json.loads(
                        self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    ),
                }
            )
            data = json.dumps(
                {"content": [{"type": "text", "text": "ok"}], "isError": False}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    stats: dict[str, Any] = {"forwarded_requests": 0, "rejected_requests": 0}
    lock = threading.Lock()

    with socketserver.TCPServer(("127.0.0.1", 0), UpstreamHandler) as upstream:
        up_port = upstream.server_address[1]
        up_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        up_thread.start()
        try:
            proxy_cls = start_proxy._make_handler(
                f"http://127.0.0.1:{up_port}", "grader-tok", "engine-tok", stats, lock
            )
            with socketserver.TCPServer(("127.0.0.1", 0), proxy_cls) as proxy:
                px_port = proxy.server_address[1]
                px_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
                px_thread.start()
                try:
                    # Missing token -> 404.
                    bad_req = urllib.request.Request(
                        f"http://127.0.0.1:{px_port}/harness/call",
                        data=b'{"name":"game_status","arguments":{}}',
                        headers={"Content-Type": "application/json"},
                    )
                    with pytest.raises(urllib.error.HTTPError) as exc_info:
                        urllib.request.urlopen(bad_req, timeout=10)
                    assert exc_info.value.code == 404
                    assert stats["rejected_requests"] == 1

                    # GET -> 404.
                    with pytest.raises(urllib.error.HTTPError) as exc_info:
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{px_port}/harness/call", timeout=10
                        )
                    assert exc_info.value.code == 404

                    # Valid grader_token -> swaps to engine_token and forwards.
                    good_req = urllib.request.Request(
                        f"http://127.0.0.1:{px_port}/harness/call",
                        data=b'{"name":"game_status","arguments":{}}',
                        headers={
                            "Content-Type": "application/json",
                            "X-Harness-Token": "grader-tok",
                        },
                    )
                    with urllib.request.urlopen(good_req, timeout=10) as resp:
                        assert json.load(resp)["isError"] is False
                    assert stats["forwarded_requests"] == 1
                    assert upstream_seen == [
                        {
                            "path": "/harness/call",
                            "token": "engine-tok",
                            "body": {"name": "game_status", "arguments": {}},
                        }
                    ]
                finally:
                    proxy.shutdown()
        finally:
            upstream.shutdown()

    # Upstream closed -> proxy returns structured infrastructure_error.
    dead_cls = start_proxy._make_handler(
        "http://127.0.0.1:1", "grader-tok", "engine-tok", stats, lock
    )
    with socketserver.TCPServer(("127.0.0.1", 0), dead_cls) as dead_proxy:
        px_port = dead_proxy.server_address[1]
        px_thread = threading.Thread(target=dead_proxy.serve_forever, daemon=True)
        px_thread.start()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{px_port}/harness/call",
                data=b'{"name":"game_status","arguments":{}}',
                headers={"Content-Type": "application/json", "X-Harness-Token": "grader-tok"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.load(resp)
            assert "Go engine service unreachable" in body["infrastructure_error"]
        finally:
            dead_proxy.shutdown()


@pytest.mark.asyncio
async def test_katago_engine_forced_search_and_error_handling(tmp_path: Path) -> None:
    import sys

    fake_bin = tmp_path / "fake_katago.py"
    fake_bin.write_text(
        (
            f"#!{sys.executable}\n"
            "import json, sys\n"
            "for raw in sys.stdin:\n"
            "    q = json.loads(raw)\n"
            "    if q.get('moves') == [['B', 'BAD']]:\n"
            "        print(json.dumps({'id': q['id'], 'error': 'illegal move', 'field': 'moves'}), flush=True)\n"
            "        continue\n"
            "    allow = q.get('allowMoves') or []\n"
            "    move = allow[0]['moves'][0] if allow else 'Q16'\n"
            "    print(json.dumps({'id': q['id'], 'rootInfo': {'currentPlayer': 'B', 'winrate': 0.8}, "
            "'moveInfos': [{'move': move, 'order': 0, 'winrate': 0.8, 'scoreLead': 5.0}]}), flush=True)\n"
        ),
        encoding="utf-8",
    )
    fake_bin.chmod(0o755)
    net = tmp_path / "net.bin.gz"
    net.write_bytes(b"")

    engine = katago_api.KataGoEngine(str(fake_bin), str(net), max_visits=400, work_dir=tmp_path)
    cfg = engine.config_file.read_text(encoding="utf-8")
    assert "forDeterministicTesting = true" in cfg
    assert "useFP16 = false" in cfg
    try:
        await engine.start()
        forced = await engine.analyse([["B", "D4"]], visits=300, force=("W", "Q4"))
        assert forced["moveInfos"][0]["move"] == "Q4"
        with pytest.raises(katago_api.EngineError, match="illegal move"):
            await engine.analyse([["B", "BAD"]])
    finally:
        await engine.close()
