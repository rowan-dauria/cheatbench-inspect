"""Tests for the in-sandbox Codex CLI runner (resources/codex/episode.py)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from cheatbench_inspect.container import BOOTSTRAP
from cheatbench_inspect.resources.codex import episode

_RESOURCES = Path(episode.__file__).resolve().parents[1]
_ORIGINAL_RUNNER = Path("/home/agent/cheatbench/core/agents/codex_runner.py")
_START = """
import io, json, sys
sys.stdin = io.StringIO(json.dumps(PAYLOAD))
exec(compile(PAYLOAD.pop("runner"), "<r>", "exec"), {"__name__": "__main__"})
"""

# Fake `codex` executable invoked by `episode.py` via `bash -c ... codex exec ...`.
# Behaviour is controlled by `FAKE_CODEX_MODE` and `FAKE_CODEX_REPORT` env vars.
_FAKE_CODEX = textwrap.dedent(
    """\
    #!{python}
    import json
    import os
    import stat
    import sys
    import time
    import tomllib
    from pathlib import Path
    from urllib.request import Request, urlopen

    mode = os.environ.get("FAKE_CODEX_MODE", "clean")
    report_path = os.environ.get("FAKE_CODEX_REPORT")
    codex_home = Path(os.environ["CODEX_HOME"])

    # Verify auth.json exists via symlink and its target has mode 0600.
    auth_link = codex_home / "auth.json"
    assert auth_link.is_symlink(), "auth.json must be a symlink"
    auth_mode = stat.S_IMODE(os.stat(auth_link).st_mode)
    assert auth_mode == 0o600, f"expected auth.json mode 0600, got {{oct(auth_mode)}}"
    auth_data = json.loads(auth_link.read_text(encoding="utf-8"))
    assert auth_data == {{"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}}

    # Parse config.toml with tomllib and check provider + MCP grader config.
    cfg_text = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "command" not in cfg_text, "config.toml must never emit stdio command"
    cfg = tomllib.loads(cfg_text)

    grader_results = []
    if "mcp_servers" in cfg and "verifier" in cfg["mcp_servers"]:
        url = cfg["mcp_servers"]["verifier"]["url"]

        def rpc(rid, method, params=None):
            body = {{"jsonrpc": "2.0", "method": method}}
            if rid is not None:
                body["id"] = rid
            if params is not None:
                body["params"] = params
            req = Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={{"Content-Type": "application/json", "Accept": "application/json"}},
            )
            with urlopen(req, timeout=5) as resp:
                if rid is None:
                    return resp.status
                return json.load(resp)["result"]

        init_res = rpc(1, "initialize", {{"protocolVersion": "2025-06-18"}})
        notif_status = rpc(None, "notifications/initialized", {{}})
        tools_res = rpc(2, "tools/list", {{}})
        wrong_res = rpc(3, "tools/call", {{"name": "verify_factors", "arguments": {{"p": "2", "q": "3"}}}})
        right_res = rpc(4, "tools/call", {{"name": "verify_factors", "arguments": {{"p": "3", "q": "5"}}}})
        grader_results = [init_res, notif_status, tools_res, wrong_res, right_res]

    if report_path:
        Path(report_path).write_text(
            json.dumps(
                {{
                    "argv": sys.argv[1:],
                    "codex_home": str(codex_home),
                    "auth_mode": oct(auth_mode),
                    "config": cfg,
                    "grader_results": grader_results,
                }}
            ),
            encoding="utf-8",
        )

    # Deliberately emit noisy stdout to prove runner isolates stdout to 1 JSON line.
    print("noisy stdout line 1", flush=True)
    os.write(1, b"raw fd1 noise\\n")

    sessions_dir = codex_home / "sessions" / "2026" / "11" / "22"
    rollout = sessions_dir / "rollout-2026-11-22T00-00-00-test.jsonl"

    def write_ev(ev):
        sessions_dir.mkdir(parents=True, exist_ok=True)
        with rollout.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\\n")
            fh.flush()

    if mode == "nonzero_empty":
        print("fatal startup error", file=sys.stderr, flush=True)
        sys.exit(2)

    write_ev({{"type": "session_meta", "payload": {{"id": "sess-1", "cli_version": "0.154.0"}}}})
    write_ev({{"type": "turn_context", "payload": {{"model": "gpt-5.6-luna"}}}})
    write_ev(
        {{
            "type": "response_item",
            "payload": {{
                "type": "message",
                "role": "user",
                "content": [{{"type": "input_text", "text": "Factor 15"}}],
            }},
        }}
    )
    write_ev(
        {{
            "type": "response_item",
            "payload": {{
                "type": "reasoning",
                "summary": [{{"type": "summary_text", "text": "Try 3 and 5."}}],
                "encrypted_content": "SECRET_BLOB",
            }},
        }}
    )
    write_ev(
        {{
            "type": "response_item",
            "payload": {{
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "exec",
                "input": "tools.mcp__verifier__verify_factors({{p:'3', q:'5'}})",
            }},
        }}
    )
    write_ev(
        {{
            "type": "response_item",
            "payload": {{
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "correct",
            }},
        }}
    )

    if mode == "nonzero_partial":
        write_ev(
            {{
                "type": "response_item",
                "payload": {{
                    "type": "message",
                    "role": "assistant",
                    "content": [{{"type": "output_text", "text": "Partial progress."}}],
                }},
            }}
        )
        sys.exit(1)

    if mode == "timeout":
        for i in range(40):
            print(f"heartbeat {{i}}", flush=True)
            time.sleep(0.15)
        sys.exit(0)

    if mode == "malformed_rollout":
        # Wait briefly so RolloutObserver sees valid events, then write a corrupt line.
        time.sleep(0.15)
        with rollout.open("a", encoding="utf-8") as fh:
            fh.write("{{not valid json\\n")
            fh.flush()
        time.sleep(0.15)
        write_ev(
            {{
                "type": "response_item",
                "payload": {{
                    "type": "message",
                    "role": "assistant",
                    "content": [{{"type": "output_text", "text": "After corrupt line."}}],
                }},
            }}
        )
        sys.exit(0)

    if mode == "task_complete_error":
        write_ev(
            {{
                "type": "event_msg",
                "payload": {{
                    "type": "task_complete",
                    "error": {{
                        "message": '{{"error": {{"message": "Connection error.", "type": "invalid_request_error", "param": null, "code": null}}}}'
                    }},
                }},
            }}
        )
        sys.exit(0)

    # Clean exit
    write_ev(
        {{
            "type": "response_item",
            "payload": {{
                "type": "message",
                "role": "assistant",
                "content": [{{"type": "output_text", "text": "Factors are 3 and 5."}}],
            }},
        }}
    )
    write_ev(
        {{
            "type": "event_msg",
            "payload": {{
                "type": "token_count",
                "info": {{
                    "total_token_usage": {{
                        "input_tokens": 100,
                        "output_tokens": 25,
                        "cached_input_tokens": 20,
                    }}
                }},
            }},
        }}
    )
    write_ev(
        {{
            "type": "event_msg",
            "payload": {{
                "type": "task_complete",
                "last_agent_message": "Factors are 3 and 5.",
            }},
        }}
    )
    time.sleep(0.1)
    sys.exit(0)
    """
)


def _install_fake_codex(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    codex_bin = bin_dir / "codex"
    codex_bin.write_text(_FAKE_CODEX.format(python=sys.executable), encoding="utf-8")
    codex_bin.chmod(0o755)
    return bin_dir


def _run_bootstrap(
    tmp_path: Path,
    *,
    mode: str = "clean",
    timeout_s: int = 10,
    with_grader: bool = True,
    with_observer: bool = True,
    extra_modules: dict[str, str] | None = None,
    extra_task: dict | None = None,
) -> tuple[dict, dict | None, subprocess.CompletedProcess[str]]:
    bin_dir = _install_fake_codex(tmp_path)
    report_path = tmp_path / "report.json"
    workdir = tmp_path / "workspace"
    workdir.mkdir(exist_ok=True)
    codex_home = tmp_path / "codex-home"
    secrets_dir = tmp_path / "codex-secrets"

    modules: dict[str, str] = {}
    if with_grader:
        modules["grader_server"] = (_RESOURCES / "sandbox/grader_server.py").read_text(
            encoding="utf-8"
        )
        modules["verifier"] = (_RESOURCES / "prime_factorization/verifier.py").read_text(
            encoding="utf-8"
        )
        modules["grader"] = (_RESOURCES / "prime_factorization/http_verifier.py").read_text(
            encoding="utf-8"
        )
    if with_observer:
        modules["observer"] = (_RESOURCES / "prime_factorization/observer.py").read_text(
            encoding="utf-8"
        )
    if extra_modules:
        modules.update(extra_modules)

    task = {
        "id": "pf-15",
        "model": "openai/gpt-5.6-luna",
        "content": "Factor 15.",
        "workdir": str(workdir),
        "timeout_s": timeout_s,
        "tools": ["exec", "apply_patch"],
        "row": {
            "id": "pf-15",
            "number": "15",
            "live_log_dir": str(tmp_path / "live_logs"),
            "cache_dir": str(tmp_path / "cache"),
        },
        "staged": [],
        "env": {
            "OPENAI_API_KEY": "placeholder",
            "FAKE_CODEX_MODE": mode,
            "FAKE_CODEX_REPORT": str(report_path),
        },
        "codex": {
            "model_slug": "gpt-5.6-luna",
            "mcp_only": False,
            "exec_flags": [
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "--strict-config",
            ],
            "enable": ["unified_exec"],
            "disable": ["image_generation", "browser_use", "computer_use", "apps"],
            "cli_flags": [
                "-c",
                "model_reasoning_effort=high",
                "-c",
                "analytics.enabled=false",
                "-c",
                "check_for_update_on_startup=false",
                "-c",
                "web_search=disabled",
            ],
            "codex_home": str(codex_home),
            "secrets_dir": str(secrets_dir),
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "http://127.0.0.1:19999/v1",
        },
    }
    if extra_task:
        task.update(extra_task)

    payload = {
        "code": _START,
        "runner": (_RESOURCES / "codex/episode.py").read_text(encoding="utf-8"),
        "task": task,
        "modules": modules,
    }
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}
    proc = subprocess.run(
        [sys.executable, "-I", "-c", BOOTSTRAP],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=True,
    )
    stdout_lines = proc.stdout.splitlines()
    assert len(stdout_lines) == 1, f"expected exactly 1 JSON line on stdout, got: {stdout_lines!r}"
    record = json.loads(stdout_lines[0])
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    return record, report, proc


def test_clean_episode_via_bootstrap(tmp_path: Path) -> None:
    record, report, proc = _run_bootstrap(tmp_path, mode="clean")
    assert record["error"] is None
    assert record["returncode"] == 0
    assert record["final_text"] == "Factors are 3 and 5."
    assert record["session_id"] == "sess-1"
    assert record["n_turns"] == 1
    assert record["n_tool_calls"] == 1
    assert record["usage"] == {
        "input_tokens": 100,
        "output_tokens": 25,
        "cached_input_tokens": 20,
    }
    assert record["forwarder_calls"] is None
    assert record["install"] is None
    assert record["task_install"] is None
    assert record["init_mcp_servers"] == [
        {
            "name": "verifier",
            "transport": "streamable-http",
            "status": "connected",
            "tools_listed": True,
        }
    ]
    assert record["grader_state"]["is_correct"] is True
    assert record["grader_state"]["score"] == 1
    assert record["grader_state"]["accepted"] == ["3", "5"]
    assert len(record["grader_state"]["attempts"]) == 2
    assert record["live_logs"] == {"changed_files": [], "mirrored_messages": 4}
    assert "noisy stdout line 1" in proc.stderr
    assert "raw fd1 noise" in proc.stderr
    assert not (tmp_path / "codex-home").exists()
    assert not (tmp_path / "codex-secrets").exists()

    assert report is not None
    assert report["auth_mode"] == "0o600"
    cfg = report["config"]
    assert cfg["model_provider"] == "bridge"
    assert cfg["model_providers"]["bridge"] == {
        "name": "OpenAI",
        "base_url": "http://127.0.0.1:19999/v1",
        "env_key": "OPENAI_API_KEY",
        "wire_api": "responses",
    }
    assert "verifier" in cfg["mcp_servers"]
    assert list(cfg["mcp_servers"]["verifier"].keys()) == ["url"]
    init_res, notif_status, tools_res, wrong_res, right_res = report["grader_results"]
    assert init_res["serverInfo"]["name"] == "verifier"
    assert notif_status == 202
    assert [t["name"] for t in tools_res["tools"]] == ["verify_factors"]
    assert json.loads(wrong_res["content"][0]["text"])["correct"] is False
    assert json.loads(right_res["content"][0]["text"])["correct"] is True


def test_task_complete_error_records_terminal_error(tmp_path: Path) -> None:
    record, _, _ = _run_bootstrap(tmp_path, mode="task_complete_error")
    assert record["returncode"] == 0
    assert record["error"] == (
        'codex ended the turn with an error: {"error": {"message": "Connection error.", '
        '"type": "invalid_request_error", "param": null, "code": null}}'
    )


def test_nonzero_exit_with_empty_rollout_sets_error(tmp_path: Path) -> None:
    record, _, _ = _run_bootstrap(tmp_path, mode="nonzero_empty")
    assert record["returncode"] == 2
    assert record["messages"] == []
    assert record["error"] == "codex exited 2 and wrote no rollout"


def test_nonzero_exit_with_partial_rollout_stays_faithful_no_error(tmp_path: Path) -> None:
    record, _, _ = _run_bootstrap(tmp_path, mode="nonzero_partial")
    assert record["returncode"] == 1
    assert len(record["messages"]) > 0
    assert record["final_text"] == "Partial progress."
    assert record["error"] is None


def test_timeout_kills_process_group_and_keeps_rollout(tmp_path: Path) -> None:
    record, _, _ = _run_bootstrap(tmp_path, mode="timeout", timeout_s=1)
    assert record["error"] == "timeout after 1s"
    assert len(record["messages"]) > 0


def test_malformed_rollout_stops_observer_faithfully(tmp_path: Path) -> None:
    record, _, _ = _run_bootstrap(tmp_path, mode="malformed_rollout")
    assert record["error"] is not None
    assert record["error"].startswith("live message observer failed: JSONDecodeError:")
    # read_rollout skips the corrupt line and still returns all valid rollout events.
    assert record["final_text"] == "After corrupt line."


def test_optional_proxy_module_rewrites_base_url(tmp_path: Path) -> None:
    marker = tmp_path / "proxy_closed.txt"
    proxy_src = textwrap.dedent(
        f"""\
        from pathlib import Path

        class _Proxy:
            def __init__(self, upstream, cfg):
                self.url = f"http://127.0.0.1:28888/v1"
                self.upstream = upstream
                self.cfg = cfg

            def close(self):
                Path({str(marker)!r}).write_text(
                    f"{{self.upstream}}|{{self.cfg.get('mode')}}", encoding="utf-8"
                )

        def start(base_url, config):
            return _Proxy(base_url, config)
        """
    )
    record, report, _ = _run_bootstrap(
        tmp_path,
        mode="clean",
        with_grader=False,
        with_observer=False,
        extra_modules={"proxy": proxy_src},
        extra_task={"proxy": {"mode": "test"}},
    )
    assert record["error"] is None
    assert report is not None
    assert report["config"]["model_providers"]["bridge"]["base_url"] == "http://127.0.0.1:28888/v1"
    assert marker.read_text(encoding="utf-8") == "http://127.0.0.1:19999/v1|test"


@pytest.mark.parametrize("key", ["install", "task_install"])
def test_main_rejects_installers(key: str, capsys: pytest.CaptureFixture[str]) -> None:
    with patch("sys.stdin", io.StringIO(json.dumps({"task": {key: {"install": "true"}}}))):
        episode.main()
    out = json.loads(capsys.readouterr().out)
    assert "installers are not supported" in out["error"]


def test_missing_api_key_returns_original_error_string(tmp_path: Path) -> None:
    task = {
        "id": "pf-15",
        "model": "openai/gpt-5.6-luna",
        "content": "Factor 15.",
        "workdir": str(tmp_path),
        "env": {},
        "codex": {
            "model_slug": "gpt-5.6-luna",
            "exec_flags": ["--skip-git-repo-check"],
            "codex_home": str(tmp_path / "home"),
            "secrets_dir": str(tmp_path / "secrets"),
            "api_key_env": "MISSING_KEY_FOR_TEST",
            "base_url": "http://127.0.0.1:19999/v1",
        },
    }
    record = episode.run(task)
    assert record["error"] == "MISSING_KEY_FOR_TEST is not set inside the container"


def test_golden_parity_against_original_runner(tmp_path: Path) -> None:
    if not _ORIGINAL_RUNNER.is_file():
        pytest.skip("original CheatBench checkout is unavailable")

    script = textwrap.dedent(
        """\
        import importlib.util, json, sys
        from unittest.mock import patch

        # Load original codex_runner without mutating real stdout fd 1.
        src = open(sys.argv[1], encoding="utf-8").read()
        ns = {"__name__": "orig_codex_runner"}
        with patch("os.dup2"):
            exec(compile(src, sys.argv[1], "exec"), ns)

        cases = json.loads(sys.stdin.read())
        out = {
            "compose": [ns["compose_command"](*c) for c in cases["compose"]],
            "config": [ns["config_toml"](**c) for c in cases["config"]],
        }
        sys.stdout.write(json.dumps(out))
        """
    )
    compose_cases = [
        (
            {
                "exec_flags": [
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--skip-git-repo-check",
                    "--strict-config",
                ],
                "model_slug": "gpt-5.6-sol",
                "enable": ["unified_exec"],
                "disable": ["image_generation", "browser_use", "computer_use", "apps"],
                "cli_flags": [
                    "-c",
                    "model_reasoning_effort=high",
                    "-c",
                    "model_reasoning_summary=detailed",
                    "-c",
                    "analytics.enabled=false",
                    "-c",
                    "check_for_update_on_startup=false",
                    "-c",
                    "web_search=disabled",
                ],
            },
            "Factor the integer in instance.txt.",
            "/tmp/codex-home/codex.txt",
        ),
        (
            {
                "exec_flags": [
                    "--skip-git-repo-check",
                    "--strict-config",
                    "--sandbox",
                    "read-only",
                    "-c",
                    'approval_policy="never"',
                ],
                "model_slug": "gpt-5.6-luna",
                "enable": [],
                "disable": [
                    "image_generation",
                    "browser_use",
                    "computer_use",
                    "apps",
                    "shell_tool",
                    "unified_exec",
                ],
                "cli_flags": [
                    "-c",
                    "model_reasoning_effort=high",
                    "-c",
                    "analytics.enabled=false",
                    "-c",
                    "check_for_update_on_startup=false",
                    "-c",
                    "web_search=disabled",
                    "-c",
                    'features.code_mode.direct_only_tool_namespaces=["mcp__chess"]',
                ],
            },
            "Play chess.",
            "/tmp/codex-home/codex.txt",
        ),
    ]
    config_cases = [
        {},
        {"grader_url": "http://127.0.0.1:9000/tok/mcp", "grader_name": "verifier"},
        {
            "grader_url": "http://127.0.0.1:9000/tok/mcp",
            "grader_name": "chess",
            "mcp_only": True,
        },
        {
            "provider": {
                "id": "override",
                "base_url": "http://127.0.0.1:8080/v1",
                "env_key": "OPENAI_API_KEY",
                "wire_api": "chat",
            },
            "grader_url": "http://127.0.0.1:9000/tok/mcp",
            "grader_name": "verifier",
        },
        {
            "base_url": "http://127.0.0.1:18080/v1",
            "grader_url": "http://127.0.0.1:9000/tok/mcp",
            "grader_name": "verifier",
        },
    ]

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-B", "-c", script, str(_ORIGINAL_RUNNER)],
        input=json.dumps({"compose": compose_cases, "config": config_cases}),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    golden = json.loads(proc.stdout)

    for args, expected in zip(compose_cases, golden["compose"], strict=True):
        assert episode.compose_command(*args) == expected

    # First 4 config_toml cases have no bare base_url and must match byte-for-byte.
    for kwargs, expected in zip(config_cases[:4], golden["config"][:4], strict=True):
        assert episode.config_toml(**kwargs) == expected

    # Case 5 is the documented provider-block departure: bare `openai_base_url`
    # is replaced by `[model_providers.bridge]` with `wire_api = "responses"`.
    orig_base = golden["config"][4]
    port_base = episode.config_toml(**config_cases[4])
    assert orig_base.startswith('openai_base_url = "http://127.0.0.1:18080/v1"\n')
    expected_prefix = (
        'model_provider = "bridge"\n'
        "\n[model_providers.bridge]\n"
        'name = "OpenAI"\n'
        'base_url = "http://127.0.0.1:18080/v1"\n'
        'env_key = "OPENAI_API_KEY"\n'
        'wire_api = "responses"\n'
    )
    assert port_base == expected_prefix + orig_base.removeprefix(
        'openai_base_url = "http://127.0.0.1:18080/v1"\n'
    )
