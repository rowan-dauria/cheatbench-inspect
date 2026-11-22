"""End-to-end Docker test for Codex Responses reasoning replay and <content-internal> normalization.

Installs real Codex 0.154.0 via inspect_swe 0.2.72 into the port's sandbox
Dockerfile `task` stage, runs `resources/codex/episode.py` with
`resources/codex/proxy.py`, and routes through the real `sandbox_agent_bridge`
(with `responses_reasoning_filter()`) and real `OpenRouterAPI(responses_api=True)`
pointed at a local loopback fake OpenRouter server (dummy key, offline).
"""

from __future__ import annotations

import http.server
import json
import socketserver
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from cheatbench_inspect import codex_trajectory
from cheatbench_inspect.container import AGENT_ENV, AGENT_USER, BOOTSTRAP
from cheatbench_inspect.judge import render_trajectory
from cheatbench_inspect.reasoning_replay import responses_reasoning_filter

_CODEX_VERSION = "0.154.0"
_CLI_MODEL = "openai/gpt-5.6-luna"
_PROOF_TOKEN = "proof-codex-e2e-4729"
_RESOURCES = Path(__file__).resolve().parents[2] / "src/cheatbench_inspect/resources"
_START = """
import io, json, sys
sys.stdin = io.StringIO(json.dumps(PAYLOAD))
exec(compile(PAYLOAD.pop("runner"), "<r>", "exec"), {"__name__": "__main__"})
"""


def _make_openrouter_response(
    *,
    response_id: str,
    model: str,
    output: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    reasoning_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": response_id,
        "object": "response",
        "created_at": now,
        "completed_at": now,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "metadata": {},
        "model": model,
        "output": output,
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": {
            "effort": (reasoning_cfg or {}).get("effort", "high"),
            "summary": (reasoning_cfg or {}).get("summary", "detailed"),
            "generate_summary": None,
        },
        "store": False,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": tools or [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 80,
            "output_tokens_details": {"reasoning_tokens": 30},
            "total_tokens": 1080,
        },
        "user": None,
    }


def _scripted_turns() -> list[list[dict[str, Any]]]:
    cmd_js = (
        f'const r = await tools.exec_command({{cmd:"echo {_PROOF_TOKEN} > /tmp/proof.txt && '
        f'cat /tmp/proof.txt",workdir:"/workspace",yield_time_ms:10000,max_output_tokens:1000}}); '
        f"text(r.output);\n"
    )
    turn1 = [
        # Encrypted-only reasoning item (empty summary).
        {
            "id": "rs_turn1_enc_only_11111111111111111111111111111111",
            "type": "reasoning",
            "summary": [],
            "content": None,
            "encrypted_content": (
                "gAAAAABrE2E_TURN1_ENC_ONLY_BLOB_0123456789abcdef=="
                ".eyJlbmRwb2ludF9zbHVnIjoib3BlbmFpL2dwdC01LjYtbHVuYSJ9"
            ),
            "status": "completed",
            "format": "openai-responses-v1",
        },
        # Summary + encrypted reasoning item.
        {
            "id": "rs_turn1_sum_enc_22222222222222222222222222222222",
            "type": "reasoning",
            "summary": [
                {
                    "type": "summary_text",
                    "text": (
                        "**Planning command execution**\n\n"
                        "Write the proof token to /tmp/proof.txt and read it back via exec."
                    ),
                }
            ],
            "content": None,
            "encrypted_content": (
                "gAAAAABrE2E_TURN1_SUM_ENC_BLOB_abcdef0123456789=="
                ".eyJlbmRwb2ludF9zbHVnIjoib3BlbmFpL2dwdC01LjYtbHVuYSJ9"
            ),
            "status": "completed",
            "format": "openai-responses-v1",
        },
        {
            "id": "msg_e2e_turn1_commentary",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "phase": "commentary",
            "content": [
                {
                    "type": "output_text",
                    "text": "Executing the requested shell command.",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        },
        {
            "id": "ctc_e2e_turn1_exec",
            "type": "custom_tool_call",
            "call_id": "call_e2e_001",
            "name": "exec",
            "namespace": "functions",
            "input": cmd_js,
            "status": "completed",
        },
    ]
    turn2 = [
        {
            "id": "rs_turn2_sum_enc_33333333333333333333333333333333",
            "type": "reasoning",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "**Reporting output**\n\nThe shell command returned the proof token.",
                }
            ],
            "content": None,
            "encrypted_content": (
                "gAAAAABrE2E_TURN2_SUM_ENC_BLOB_9876543210fedcba=="
                ".eyJlbmRwb2ludF9zbHVnIjoib3BlbmFpL2dwdC01LjYtbHVuYSJ9"
            ),
            "status": "completed",
            "format": "openai-responses-v1",
        },
        {
            "id": "msg_e2e_turn2_final",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": _PROOF_TOKEN,
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        },
    ]
    return [turn1, turn2]


class _FakeOpenRouterHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _FakeOpenRouterServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        req_doc = json.loads(raw.decode("utf-8"))
        with self.server.lock:
            turn_idx = len(self.server.requests)
            self.server.requests.append({"path": self.path, "body": req_doc})
            output = (
                self.server.turns[turn_idx]
                if turn_idx < len(self.server.turns)
                else self.server.turns[-1]
            )
        resp = _make_openrouter_response(
            response_id=f"gen-fake-e2e-turn{turn_idx + 1}",
            model=req_doc.get("model", _CLI_MODEL),
            output=output,
            tools=req_doc.get("tools"),
            reasoning_cfg=req_doc.get("reasoning"),
        )
        encoded = json.dumps(resp).encode("utf-8")
        self.send_response(200, "OK")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
        self.wfile.flush()


class _FakeOpenRouterServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self.turns = turns
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _FakeOpenRouterHandler)
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.server_address[0]!s}:{self.server_address[1]}/api/v1"

    def start(self) -> _FakeOpenRouterServer:
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self.shutdown()
        self.server_close()


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.dataset_download
def test_codex_responses_reasoning_and_content_internal_e2e(tmp_path: Path) -> None:
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.agent import sandbox_agent_bridge
    from inspect_ai.dataset import Sample
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_ai.solver import solver
    from inspect_ai.util import sandbox
    from inspect_swe._codex_cli.agentbinary import codex_cli_binary_source
    from inspect_swe._util.agentbinary import ensure_agent_binary_installed

    compose = _RESOURCES / "sandbox/codex-compose.yaml"
    subprocess.run(["docker", "compose", "-f", str(compose), "build"], check=True)

    turns = _scripted_turns()
    fake = _FakeOpenRouterServer(turns).start()
    captured: dict[str, Any] = {}
    try:
        model_obj = get_model(
            f"openrouter/{_CLI_MODEL}",
            base_url=fake.base_url,
            api_key="dummy-openrouter-key",
            responses_api=True,
            config=GenerateConfig(reasoning_effort="high", reasoning_summary="detailed"),
        )

        @solver
        def run_episode():
            async def solve(state, generate):
                del generate
                box = sandbox()
                codex_binary = await ensure_agent_binary_installed(
                    codex_cli_binary_source(), _CODEX_VERSION, AGENT_USER, box
                )
                install_res = await box.exec(
                    [
                        "sh",
                        "-c",
                        'chmod -R a+rX "$(dirname "$(dirname "$1")")" && ln -sf "$1" /usr/local/bin/codex',
                        "sh",
                        codex_binary,
                    ],
                    user="root",
                )
                assert install_res.success, install_res.stderr

                async with sandbox_agent_bridge(
                    model_aliases={_CLI_MODEL: model_obj},
                    filter=responses_reasoning_filter(),
                    forward_generation_config=False,
                ) as bridge:
                    payload = {
                        "code": _START,
                        "runner": (_RESOURCES / "codex/episode.py").read_text(encoding="utf-8"),
                        "task": {
                            "id": "codex-e2e-1",
                            "model": _CLI_MODEL,
                            "content": state.input_text,
                            "workdir": "/workspace",
                            "timeout_s": 180,
                            "env": {"OPENAI_API_KEY": "placeholder-key"},
                            "codex": {
                                "model_slug": _CLI_MODEL,
                                "exec_flags": [
                                    "--dangerously-bypass-approvals-and-sandbox",
                                    "--skip-git-repo-check",
                                ],
                                "cli_flags": [
                                    "-c",
                                    "analytics.enabled=false",
                                    "-c",
                                    "web_search=disabled",
                                ],
                                "codex_home": "/tmp/codex-e2e-home",
                                "secrets_dir": "/tmp/codex-e2e-secrets",
                                "api_key_env": "OPENAI_API_KEY",
                                "base_url": f"http://127.0.0.1:{bridge.port}/v1",
                            },
                        },
                        "modules": {
                            "proxy": (_RESOURCES / "codex/proxy.py").read_text(encoding="utf-8"),
                        },
                    }
                    res = await box.exec(
                        ["python3", "-I", "-c", BOOTSTRAP],
                        input=json.dumps(payload),
                        timeout=240,
                        user=AGENT_USER,
                        env=AGENT_ENV,
                    )
                assert res.success, f"runner failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
                captured["record"] = json.loads(res.stdout.splitlines()[-1])
                captured["proof_text"] = (await box.read_file("/tmp/proof.txt")).strip()
                return state

            return solve

        [log] = inspect_eval(
            Task(
                dataset=[
                    Sample(
                        id="codex-e2e-1",
                        input="Run the requested shell command and reply with its exact output.",
                    )
                ],
                solver=run_episode(),
                sandbox=("docker", str(compose)),
            ),
            model=model_obj,
            log_dir=str(tmp_path),
            display="none",
        )
        assert log.status == "success", log.error
        assert log.samples
        assert log.samples[0].error is None, log.samples[0].error

        record = captured["record"]
        assert record["error"] is None, record
        assert record["returncode"] == 0
        assert captured["proof_text"] == _PROOF_TOKEN
        assert record["final_text"] == _PROOF_TOKEN
        assert len(fake.requests) == 2

        # 1. Rollout has native reasoning items matching all scripted Turn-1 & Turn-2 items.
        rollout_events = record["messages"]
        rollout_json = json.dumps(rollout_events)
        assert "<think" not in rollout_json
        assert "<content-internal" not in rollout_json
        assert "<think" not in record["final_text"]
        assert "<content-internal" not in record["final_text"]

        response_items = [
            ev["payload"]
            for ev in rollout_events
            if isinstance(ev, dict) and ev.get("type") == "response_item"
        ]
        rollout_reasoning = [p for p in response_items if p.get("type") == "reasoning"]
        expected_reasoning = [
            item for turn in turns for item in turn if item.get("type") == "reasoning"
        ]
        assert len(rollout_reasoning) == len(expected_reasoning) == 3
        for exp, got in zip(expected_reasoning, rollout_reasoning, strict=True):
            assert got.get("encrypted_content") == exp["encrypted_content"]
            assert got.get("summary") == exp["summary"]

        # 2. Rollout assistant messages carry native phase (and id) with clean text.
        rollout_asst = [
            p for p in response_items if p.get("type") == "message" and p.get("role") == "assistant"
        ]
        assert len(rollout_asst) == 2
        assert rollout_asst[0]["id"] == "msg_e2e_turn1_commentary"
        assert rollout_asst[0]["phase"] == "commentary"
        assert rollout_asst[0]["content"][0]["text"] == "Executing the requested shell command."
        assert rollout_asst[1]["id"] == "msg_e2e_turn2_final"
        assert rollout_asst[1]["phase"] == "final_answer"
        assert rollout_asst[1]["content"][0]["text"] == _PROOF_TOKEN

        # 3. exec ran cleanly (custom_tool_call_output, not aborted).
        custom_outputs = [p for p in response_items if p.get("type") == "custom_tool_call_output"]
        assert len(custom_outputs) == 1
        output_blob = json.dumps(custom_outputs[0])
        assert _PROOF_TOKEN in output_blob
        assert "aborted" not in output_blob.lower()

        # 4. Turn-2 upstream request carries Turn-1 reasoning items byte-identical and
        #    assistant message id/phase restored as the bridge would send without proxy.
        turn2_input = fake.requests[1]["body"]["input"]
        turn2_reasoning = [
            it for it in turn2_input if isinstance(it, dict) and it.get("type") == "reasoning"
        ]
        expected_turn1_reasoning = [it for it in turns[0] if it.get("type") == "reasoning"]
        assert len(turn2_reasoning) == len(expected_turn1_reasoning) == 2
        for exp, got in zip(expected_turn1_reasoning, turn2_reasoning, strict=True):
            assert got.get("id") == exp["id"]
            assert got.get("encrypted_content") == exp["encrypted_content"]
            assert got.get("summary") == exp["summary"]

        turn2_asst = [
            it
            for it in turn2_input
            if isinstance(it, dict)
            and it.get("type") == "message"
            and it.get("role") == "assistant"
        ]
        assert len(turn2_asst) == 1
        assert turn2_asst[0]["id"] == "msg_e2e_turn1_commentary"
        assert turn2_asst[0]["phase"] == "commentary"
        assert turn2_asst[0]["content"][0]["text"] == "Executing the requested shell command."

        # 5. codex_trajectory.to_trajectory + render_trajectory renders THINKING: lines and no tags.
        trajectory = codex_trajectory.to_trajectory(
            record, cli_version=_CODEX_VERSION, cli_model=_CLI_MODEL
        )
        rendered = render_trajectory(trajectory)
        assert "THINKING: **Planning command execution**" in rendered
        assert "THINKING: **Reporting output**" in rendered
        assert f"FINAL MESSAGE: {_PROOF_TOKEN}" in rendered
        assert "<think" not in rendered
        assert "<content-internal" not in rendered
    finally:
        fake.close()
