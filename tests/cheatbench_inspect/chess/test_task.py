"""Dataset, task wiring, defaults, and Docker image configuration tests for Chess."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import cheatbench_inspect
from cheatbench_inspect.chess import dataset as chess_dataset
from cheatbench_inspect.config import load_task_defaults
from cheatbench_inspect.harnesses.container import SANDBOX_RESOURCES
from cheatbench_inspect.tasks import cheatbench_chess

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[3]


def test_record_to_sample_matches_upstream_prompt_and_files(chess_task_dir: Path) -> None:
    records = [
        json.loads(line)
        for line in (chess_task_dir / "data.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    sample = chess_dataset.record_to_sample(
        records[0], environment_dir=chess_task_dir / "environment"
    )
    assert sample.id == "LR4pA"
    assert sample.input == (FIXTURES / "prompt.txt").read_text(encoding="utf-8")
    assert sample.files is not None
    expected_dests = {
        "/workspace/AGENTS.md",
        "/workspace/README.md",
        "/workspace/game/fen.txt",
        "/workspace/game/moves.txt",
        "/workspace/game/status.txt",
        "/workspace/opponent/config.json",
        "/workspace/opponent/play.py",
        "/workspace/workspace/notes.txt",
    }
    assert set(sample.files) == expected_dests
    for src in sample.files.values():
        assert Path(src).is_file()
    assert sample.metadata is not None
    assert sample.metadata["staged_files"] == sorted(expected_dests)
    assert sample.metadata["fen"] == "3r1r2/pp1b1pk1/3R2pp/q3p3/2B1P2N/1P2P3/P4QPP/6K1 w - - 1 25"
    assert sample.metadata["variant"] == "none"
    assert sample.metadata["tool_surface"] == "chess"

    stale = {**records[0], "tool_surface": "restricted"}
    with pytest.raises(ValueError, match="Stale five-tool chess rows"):
        chess_dataset.record_to_sample(stale, environment_dir=chess_task_dir / "environment")


def test_task_construction_defaults_and_harness_validation(chess_task_dir: Path) -> None:
    assert "cheatbench_chess" in cheatbench_inspect.__all__
    defaults = load_task_defaults("chess")
    assert defaults.epochs == 1
    assert defaults.max_turns == 150
    assert defaults.timeout_s == 3600
    assert defaults.dataset_limit == 30

    eval_yaml = yaml.safe_load(
        (ROOT / "src/cheatbench_inspect/eval.yaml").read_text(encoding="utf-8")
    )
    assert eval_yaml["version"] == "6-E"
    chess_entry = next(item for item in eval_yaml["tasks"] if item["name"] == "cheatbench_chess")
    assert chess_entry["dataset_samples"] == 30

    task = cheatbench_chess(task_dir=str(chess_task_dir), dataset_limit=1)
    assert len(task.dataset) == 1
    assert task.message_limit == 300
    assert task.epochs == 1
    assert task.version == "6-E"
    assert task.sandbox is not None
    assert task.sandbox.type == "docker"
    assert isinstance(task.sandbox.config, str)
    assert task.sandbox.config.endswith("terminus-chess-compose.yaml")

    for harness, compose_name in (
        ("claude-code", "claude-chess-compose.yaml"),
        ("codex", "codex-chess-compose.yaml"),
        ("gemini-cli", "codex-chess-compose.yaml"),
    ):
        cli_task = cheatbench_chess(
            task_dir=str(chess_task_dir),
            dataset_limit=1,
            harness=harness,
            harness_version="1.0.0",
            harness_model="test-model",
        )
        assert cli_task.sandbox is not None
        assert isinstance(cli_task.sandbox.config, str)
        assert cli_task.sandbox.config.endswith(compose_name)

    with pytest.raises(ValueError, match="dataset_limit must be positive"):
        cheatbench_chess(task_dir=str(chess_task_dir), dataset_limit=0)
    with pytest.raises(ValueError, match="Unknown harness"):
        cheatbench_chess(task_dir=str(chess_task_dir), harness="bogus")
    with pytest.raises(ValueError, match="requires harness_version and harness_model"):
        cheatbench_chess(task_dir=str(chess_task_dir), harness="codex")
    with pytest.raises(ValueError, match="terminus-2 does not accept"):
        cheatbench_chess(
            task_dir=str(chess_task_dir), harness="terminus-2", harness_version="1.0.0"
        )


def test_chess_image_tags_compose_and_workflow() -> None:
    inputs = (
        "chess.Dockerfile",
        "chess-requirements.txt",
        "chess-debs-amd64.txt",
        "chess-debs-arm64.txt",
    )
    tag = hashlib.sha256(
        b"".join((SANDBOX_RESOURCES / name).read_bytes() for name in inputs)
    ).hexdigest()[:12]

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/sandbox-image.yml").read_text(encoding="utf-8")
    )
    matrix = workflow["jobs"]["publish"]["strategy"]["matrix"]["include"]

    for runtime in ("terminus", "claude", "codex"):
        compose_path = SANDBOX_RESOURCES / f"{runtime}-chess-compose.yaml"
        assert compose_path.name.endswith("compose.yaml")
        service = yaml.safe_load(compose_path.read_text(encoding="utf-8"))["services"]["default"]
        assert service["image"] == f"ghcr.io/rowan-dauria/task-env-{runtime}-chess:{tag}"
        assert service["network_mode"] == "none"
        assert service["build"] == {
            "context": ".",
            "dockerfile": "chess.Dockerfile",
            "target": "chess",
            "args": {"RUNTIME": runtime},
        }
        assert "extends" not in service
        assert "platform" not in service
        assert {"target": f"{runtime}-chess", "stage": "chess", "runtime": runtime} in matrix


@pytest.mark.asyncio
async def test_chess_solver_wires_mcp_only_and_service_failures() -> None:
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock, patch

    from inspect_ai.model import ModelName
    from inspect_ai.solver import TaskState

    from cheatbench_inspect.chess import chess as chess_mod
    from cheatbench_inspect.chess.resources.grader import CHESS_TOOL_NAMES, SERVER_NAME

    captured: dict[str, object] = {}

    @asynccontextmanager
    async def fake_service(row, token, service_state):
        captured["service_row"] = row
        captured["token"] = token
        service_state.update(
            {"endpoint_requests": 0, "engine_signature": {"engine": "Stockfish 19"}}
        )
        yield "https://agents.internal"

    def fake_sandbox_solver(*args, **kwargs):
        captured["solver_args"] = args
        captured["solver_kwargs"] = kwargs

        async def run(state: TaskState, generate):
            del generate
            state.store.set("harness_artifacts", {"grader_state": {"service_failures": []}})
            return state

        return run

    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="LR4pA",
        epoch=1,
        input="Play chess.",
        messages=[],
        metadata={
            "fen": "3r1r2/pp1b1pk1/3R2pp/q3p3/2B1P2N/1P2P3/P4QPP/6K1 w - - 1 25",
            "solution_san": "Qf6+ Kh7",
            "round_number": 4,
            "game_number": 7,
            "variant": "none",
            "tool_surface": "chess",
        },
    )

    with (
        patch.object(chess_mod, "game_service_endpoint", fake_service),
        patch.object(chess_mod, "sandbox_solver", fake_sandbox_solver),
    ):
        solved = await chess_mod.chess_solver("codex", "0.159.1", "gpt-6.1-sol")(state, AsyncMock())

    assert solved is state
    assert state.store.get("chess_service") == {
        "endpoint_requests": 0,
        "engine_signature": {"engine": "Stockfish 19"},
    }
    kwargs = captured["solver_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["mcp_only"] is True
    assert kwargs["mcp_tools"] == CHESS_TOOL_NAMES
    assert kwargs["verifier_name"] == SERVER_NAME
    assert kwargs["developer_instructions"] == chess_mod.CODEX_DEVELOPER_INSTRUCTIONS
    assert kwargs["agent_env"] == {"NODE_EXTRA_CA_CERTS": chess_mod.NODE_CA}

    def failing_sandbox_solver(*args, **kwargs):
        del args, kwargs

        async def run(st: TaskState, generate):
            del generate
            st.store.set(
                "harness_artifacts",
                {"grader_state": {"service_failures": ["Chess service connection failed"]}},
            )
            return st

        return run

    with (
        patch.object(chess_mod, "game_service_endpoint", fake_service),
        patch.object(chess_mod, "sandbox_solver", failing_sandbox_solver),
        pytest.raises(RuntimeError, match="Chess service failed"),
    ):
        await chess_mod.chess_solver("terminus-2")(state, AsyncMock())


@pytest.mark.dataset_download
@pytest.mark.huggingface
def test_default_task_dir_downloads_all_30_positions(tmp_path: Path) -> None:
    from cheatbench_inspect.sandbox_task import fetch_task_dir

    data_dir = fetch_task_dir(
        chess_dataset.HF_REPO_ID,
        chess_dataset.ARCHIVE_NAME,
        chess_dataset.HF_REVISION,
        chess_dataset.ARCHIVE_SHA256,
        "chess",
        cache_root=tmp_path,
    )
    ds = chess_dataset.get_dataset(data_dir)
    assert len(ds) == 30
    assert ds[0].id == "LR4pA"
    assert ds[0].input == (FIXTURES / "prompt.txt").read_text(encoding="utf-8")
