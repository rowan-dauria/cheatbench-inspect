"""Dataset, task wiring, defaults, and Docker/k8s compose configuration tests for Go."""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState

import cheatbench_inspect
from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.go import dataset as go_dataset
from cheatbench_inspect.go import go as go_mod
from cheatbench_inspect.go.resources.grader import GO_TOOL_NAMES, SERVER_NAME
from cheatbench_inspect.harnesses.container import SANDBOX_RESOURCES
from cheatbench_inspect.tasks import cheatbench_go

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[3]


def test_record_to_sample_matches_upstream_prompt_and_metadata(go_task_dir: Path) -> None:
    records = [
        json.loads(line)
        for line in (go_task_dir / "data.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    sample = go_dataset.record_to_sample(records[0], environment_dir=go_task_dir / "environment")
    assert sample.id == "ogs_69046172"
    assert sample.input == (FIXTURES / "prompt.txt").read_text(encoding="utf-8")
    assert sample.target == ""
    assert sample.files == {}
    assert sample.metadata is not None
    assert sample.metadata["staged_files"] == []
    assert sample.metadata["tool_surface"] == "go"
    assert sample.metadata["fixture"]["id"] == "ogs_69046172"
    assert sample.metadata["fixture"]["to_play"] == "B"
    assert sample.metadata["fixture"]["board_size"] == 19
    assert sample.metadata["fixture"]["komi"] == 0.5
    assert sample.metadata["fixture"]["rules"] == "japanese"
    assert len(sample.metadata["fixture"]["moves"]) == 76
    assert [t["name"] for t in sample.metadata["go_tools"]] == list(GO_TOOL_NAMES)

    bad = {**records[0], "tool_surface": "restricted"}
    with pytest.raises(ValueError, match="expected tool_surface='go'"):
        go_dataset.record_to_sample(bad, environment_dir=go_task_dir / "environment")


def test_task_construction_defaults_and_harness_validation(go_task_dir: Path) -> None:
    assert "cheatbench_go" in cheatbench_inspect.__all__
    defaults = load_task_defaults("go")
    assert defaults.epochs == 1
    assert defaults.max_turns == 150
    assert defaults.timeout_s == 3600
    assert defaults.dataset_limit == 20

    eval_yaml = yaml.safe_load(
        (ROOT / "src/cheatbench_inspect/eval.yaml").read_text(encoding="utf-8")
    )
    assert eval_yaml["version"] == "6-F"
    go_entry = next(item for item in eval_yaml["tasks"] if item["name"] == "cheatbench_go")
    assert go_entry["dataset_samples"] == 20

    task = cheatbench_go(task_dir=str(go_task_dir), dataset_limit=1)
    assert len(task.dataset) == 1
    assert task.message_limit == 300
    assert task.epochs == 1
    assert task.version == "6-F"
    assert task.sandbox is not None
    assert task.sandbox.type == "docker"
    assert isinstance(task.sandbox.config, str)
    assert task.sandbox.config.endswith("terminus-go-compose.yaml")

    for harness, compose_name in (
        ("claude-code", "claude-go-compose.yaml"),
        ("codex", "codex-go-compose.yaml"),
        ("gemini-cli", "codex-go-compose.yaml"),
    ):
        cli_task = cheatbench_go(
            task_dir=str(go_task_dir),
            dataset_limit=1,
            harness=harness,
            harness_version="1.0.0",
            harness_model="test-model",
        )
        assert cli_task.sandbox is not None
        assert isinstance(cli_task.sandbox.config, str)
        assert cli_task.sandbox.config.endswith(compose_name)

    with pytest.raises(ValueError, match="dataset_limit must be positive"):
        cheatbench_go(task_dir=str(go_task_dir), dataset_limit=0)
    with pytest.raises(ValueError, match="Unknown harness"):
        cheatbench_go(task_dir=str(go_task_dir), harness="bogus")
    with pytest.raises(ValueError, match="requires harness_version and harness_model"):
        cheatbench_go(task_dir=str(go_task_dir), harness="codex")
    with pytest.raises(ValueError, match="terminus-2 does not accept"):
        cheatbench_go(task_dir=str(go_task_dir), harness="terminus-2", harness_version="1.0.0")


def test_go_image_tags_compose_gpu_and_workflow() -> None:
    agent_inputs = (
        "go.Dockerfile",
        "go-debs-amd64.txt",
        "go-debs-arm64.txt",
    )
    agent_tag = hashlib.sha256(
        b"".join((SANDBOX_RESOURCES / name).read_bytes() for name in agent_inputs)
    ).hexdigest()[:12]

    engine_inputs = (
        "go-engine.Dockerfile",
        "go-engine-requirements.txt",
    )
    engine_tag = hashlib.sha256(
        b"".join((SANDBOX_RESOURCES / name).read_bytes() for name in engine_inputs)
    ).hexdigest()[:12]

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/sandbox-image.yml").read_text(encoding="utf-8")
    )
    matrix = workflow["jobs"]["publish"]["strategy"]["matrix"]["include"]

    assert {
        "target": "go-engine",
        "stage": "go-engine",
        "dockerfile": "go-engine.Dockerfile",
        "platforms": "linux/amd64",
    } in matrix

    for runtime in ("terminus", "claude", "codex"):
        compose_path = SANDBOX_RESOURCES / f"{runtime}-go-compose.yaml"
        assert compose_path.name.endswith("compose.yaml")
        doc = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        assert doc["networks"] == {"internal": {"driver": "bridge", "internal": True}}

        default_svc = doc["services"]["default"]
        assert default_svc["image"] == f"ghcr.io/rowan-dauria/task-env-{runtime}-go:{agent_tag}"
        assert default_svc["networks"] == ["internal"]
        assert default_svc["build"] == {
            "context": ".",
            "dockerfile": "go.Dockerfile",
            "target": "go",
            "args": {"RUNTIME": runtime},
        }
        assert "x-inspect_k8s_sandbox" not in default_svc
        assert "extends" not in default_svc
        assert "platform" not in default_svc

        engine_svc = doc["services"]["engine"]
        assert engine_svc["image"] == f"ghcr.io/rowan-dauria/task-env-go-engine:{engine_tag}"
        assert engine_svc["networks"] == ["internal"]
        assert engine_svc["build"] == {
            "context": ".",
            "dockerfile": "go-engine.Dockerfile",
            "target": "go-engine",
        }
        assert engine_svc["x-inspect_k8s_sandbox"] == {
            "resources": {
                "limits": {"nvidia.com/gpu": 1},
                "requests": {"nvidia.com/gpu": 1},
            }
        }
        assert "extends" not in engine_svc
        assert "platform" not in engine_svc

        assert {
            "target": f"{runtime}-go",
            "stage": "go",
            "runtime": runtime,
        } in matrix


@pytest.mark.asyncio
async def test_go_solver_wires_mcp_only_two_services_and_failures() -> None:
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def fake_engine(fixture, engine_token, service_state, *, timeout_s=3600):
        captured["fixture"] = fixture
        captured["engine_token"] = engine_token
        captured["timeout_s"] = timeout_s
        service_state.update(
            {
                "endpoint_requests": 0,
                "engine_signature": {"status": "exact", "live_engine": "KataGo v1.18.2 (CUDA)"},
                "infrastructure_error": None,
            }
        )
        yield "http://engine:8765"

    @asynccontextmanager
    async def fake_proxy(upstream_url, grader_token, engine_token, proxy_state):
        captured["upstream_url"] = upstream_url
        captured["grader_token"] = grader_token
        captured["proxy_engine_token"] = engine_token
        proxy_state.update({"requests": []})
        yield "https://agents.internal"

    def fake_sandbox_solver(*args, **kwargs):
        captured["solver_args"] = args
        captured["solver_kwargs"] = kwargs

        async def run(state: TaskState, generate):
            del generate
            state.store.set("harness_artifacts", {"grader_state": {"errors": []}})
            return state

        return run

    tools = json.loads((FIXTURES / "tools.json").read_text(encoding="utf-8"))
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="ogs_69046172",
        epoch=1,
        input="Play Go.",
        messages=[],
        metadata={
            "fixture": {"id": "ogs_69046172", "board_size": 19, "komi": 6.5, "rules": "japanese"},
            "go_tools": [
                {"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}
                for t in tools
            ],
            "tool_surface": "go",
        },
    )

    with (
        patch.object(go_mod, "engine_service_endpoint", fake_engine),
        patch.object(go_mod, "proxy_service_endpoint", fake_proxy),
        patch.object(go_mod, "sandbox_solver", fake_sandbox_solver),
    ):
        solved = await go_mod.go_solver("codex", "0.156.1", "gpt-6-sol")(state, AsyncMock())

    assert solved is state
    assert captured["upstream_url"] == "http://engine:8765"
    assert captured["engine_token"] == captured["proxy_engine_token"]
    assert captured["engine_token"] != captured["grader_token"]
    assert state.store.get("go_service") == {
        "endpoint_requests": 0,
        "engine_signature": {"status": "exact", "live_engine": "KataGo v1.18.2 (CUDA)"},
        "infrastructure_error": None,
    }
    assert state.store.get("go_proxy") == {"requests": []}

    kwargs = captured["solver_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["mcp_only"] is True
    assert kwargs["mcp_tools"] == GO_TOOL_NAMES
    assert kwargs["verifier_name"] == SERVER_NAME
    assert kwargs["developer_instructions"] == go_mod.CODEX_DEVELOPER_INSTRUCTIONS
    assert kwargs["agent_env"] == {"NODE_EXTRA_CA_CERTS": go_mod.NODE_CA}

    # Grader error propagates as RuntimeError.
    def failing_grader_solver(*args, **kwargs):
        del args, kwargs

        async def run(st: TaskState, generate):
            del generate
            st.store.set(
                "harness_artifacts",
                {"grader_state": {"errors": ["Go service failed: URLError"]}},
            )
            return st

        return run

    with (
        patch.object(go_mod, "engine_service_endpoint", fake_engine),
        patch.object(go_mod, "proxy_service_endpoint", fake_proxy),
        patch.object(go_mod, "sandbox_solver", failing_grader_solver),
        pytest.raises(RuntimeError, match="Go service failed"),
    ):
        await go_mod.go_solver("terminus-2")(state, AsyncMock())

    # Engine infrastructure error propagates as RuntimeError.
    @asynccontextmanager
    async def broken_engine(fixture, engine_token, service_state, *, timeout_s=3600):
        del fixture, engine_token, timeout_s
        service_state["infrastructure_error"] = "KataGo exited 1"
        yield "http://engine:8765"

    with (
        patch.object(go_mod, "engine_service_endpoint", broken_engine),
        patch.object(go_mod, "proxy_service_endpoint", fake_proxy),
        patch.object(go_mod, "sandbox_solver", fake_sandbox_solver),
        pytest.raises(RuntimeError, match="Go engine failed: KataGo exited 1"),
    ):
        await go_mod.go_solver("terminus-2")(state, AsyncMock())


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_default_task_dir_downloads_all_20_positions(tmp_path: Path) -> None:
    from cheatbench_inspect.sandbox_task import fetch_task_dir

    data_dir = fetch_task_dir(
        go_dataset.HF_REPO_ID,
        go_dataset.ARCHIVE_NAME,
        go_dataset.HF_REVISION,
        go_dataset.ARCHIVE_SHA256,
        "go",
        cache_root=tmp_path,
    )
    ds = go_dataset.get_dataset(data_dir)
    assert len(ds) == 20
    assert ds[0].id == "ogs_69046172"
    assert ds[0].input == (FIXTURES / "prompt.txt").read_text(encoding="utf-8")
