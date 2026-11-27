"""Offline unit tests for Creative Writings solver lifecycle and cancellation."""

import json
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import anyio
import pytest
from inspect_ai.model import ModelName, ModelOutput, get_model
from inspect_ai.solver import Generate, TaskState

from cheatbench_inspect.creative_writings import creative_writings, reviewer


def _state(tmp_path: Path) -> TaskState:
    ext = tmp_path / "_score.cpython-311-x86_64-linux-gnu.so"
    ext.write_bytes(b"ELF")
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id="museum_poem",
        epoch=1,
        input="Write a poem.",
        messages=[],
        metadata={
            "staged_files": ["/workspace/instruction.txt"],
            "reference_path": "/tmp/grading/ref/abc/candidate_1.md",
            "topic": "museum",
            "score_extensions": {"x86_64": str(ext)},
        },
    )


def _fake_sandbox(arch: str = "x86_64", arch_ok: bool = True) -> SimpleNamespace:
    written: dict[str, bytes | str] = {}

    async def exec_cmd(cmd: list[str]) -> SimpleNamespace:
        assert cmd == ["uname", "-m"]
        return SimpleNamespace(success=arch_ok, stdout=f"{arch}\n")

    async def write_file(path: str, content: bytes | str) -> None:
        written[path] = content

    async def read_file(path: str) -> str:
        assert path == "/tmp/grading/ref/abc/candidate_1.md"
        return "Reference poem."

    return SimpleNamespace(
        exec=exec_cmd, write_file=write_file, read_file=read_file, written=written
    )


def _instant_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reviewer,
        "asyncio",
        type("ReviewClock", (), {"Lock": reviewer.asyncio.Lock, "sleep": AsyncMock()}),
    )


async def test_reviewer_failure_cancels_episode_and_shuts_down_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _instant_sleep(monkeypatch)
    state = _state(tmp_path)
    box = _fake_sandbox()
    events: list[str] = []
    review_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None
    review_ready = anyio.Event()
    turn_after_failure_ran = False

    async def fake_sandbox_service(
        name: str,
        methods: list[Callable[..., Coroutine[Any, Any, Any]]],
        until: Callable[[], bool],
        environment: Any,
        *,
        user: str,
        started: anyio.Event,
    ) -> None:
        nonlocal review_fn
        assert name == "editorial"
        assert environment is box
        assert user == "root"
        [review_fn] = methods
        events.append("service:start")
        started.set()
        review_ready.set()
        try:
            await anyio.sleep_forever()
        except anyio.get_cancelled_exc_class():
            events.append("service:cancelled")
            raise
        finally:
            events.append("service:stop")

    @asynccontextmanager
    async def fake_endpoint(config: dict[str, Any], *, user: str) -> AsyncGenerator[str, None]:
        assert user == "root"
        assert "token" in config
        events.append("proxy:open")
        try:
            yield "http://127.0.0.1:41234/review"
        finally:
            events.append("proxy:close")

    async def fake_solve_episode(st: TaskState, gen: Generate) -> TaskState:
        nonlocal turn_after_failure_ran
        await review_ready.wait()
        assert review_fn is not None
        first = await review_fn("brief", "Reference poem.", "First draft.")
        assert first == {"verdict": "below", "feedback": "Tighten imagery."}
        events.append("agent:second_review")
        try:
            async with anyio.create_task_group() as inner:
                inner.start_soon(review_fn, "brief", "Reference poem.", "Second draft.")
                await anyio.sleep_forever()
                turn_after_failure_ran = True
        except anyio.get_cancelled_exc_class():
            events.append("agent:cancelled")
            raise
        return st

    attempts = 0

    def review_model(messages, tools, tool_choice, config):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ModelOutput.from_content(
                "mockllm/model", json.dumps({"verdict": "below", "feedback": "Tighten imagery."})
            )
        raise RuntimeError("mock outage")

    monkeypatch.setattr(creative_writings, "sandbox", lambda: box)
    monkeypatch.setattr(creative_writings, "sandbox_service", fake_sandbox_service)
    monkeypatch.setattr(creative_writings, "sandbox_service_endpoint", fake_endpoint)
    monkeypatch.setattr(creative_writings, "sandbox_solver", lambda *args: fake_solve_episode)
    monkeypatch.setattr(
        creative_writings,
        "get_model",
        lambda **kwargs: get_model("mockllm/model", custom_outputs=review_model, memoize=False),
    )

    with pytest.raises(RuntimeError, match="Reviewer failed: RuntimeError: mock outage"):
        await creative_writings.creative_writings_solver()(state, AsyncMock())

    assert not turn_after_failure_ran
    assert attempts == 4
    assert "agent:cancelled" in events
    assert "proxy:close" in events
    assert "service:stop" in events
    assert events.index("agent:cancelled") < events.index("proxy:close")
    assert state.metadata["reference_path"] == "/tmp/grading/ref/abc/candidate_1.md"
    assert state.metadata["topic"] == "museum"
    assert state.metadata["review_errors"] == ["RuntimeError: mock outage"]
    assert [
        (call["round"], call["draft"], call["verdict"], call["score"])
        for call in state.metadata["review_calls"]
    ] == [(1, "First draft.", "below", 0), (2, "Second draft.", None, None)]


async def test_reviewer_error_takes_precedence_over_secondary_unwind_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _instant_sleep(monkeypatch)
    state = _state(tmp_path)
    box = _fake_sandbox()
    review_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None

    async def fake_sandbox_service(
        name: str,
        methods: list[Callable[..., Coroutine[Any, Any, Any]]],
        until: Callable[[], bool],
        environment: Any,
        *,
        user: str,
        started: anyio.Event,
    ) -> None:
        nonlocal review_fn
        [review_fn] = methods
        started.set()
        await anyio.sleep_forever()

    @asynccontextmanager
    async def fake_endpoint(config: dict[str, Any], *, user: str) -> AsyncGenerator[str, None]:
        yield "http://127.0.0.1:41234/review"

    async def fake_solve_episode(st: TaskState, gen: Generate) -> TaskState:
        assert review_fn is not None
        try:
            await review_fn("brief", "Reference poem.", "Draft.")
            await anyio.sleep_forever()
        finally:
            raise RuntimeError("secondary harness exit")

    def failing_model(messages, tools, tool_choice, config):
        raise RuntimeError("upstream 503")

    monkeypatch.setattr(creative_writings, "sandbox", lambda: box)
    monkeypatch.setattr(creative_writings, "sandbox_service", fake_sandbox_service)
    monkeypatch.setattr(creative_writings, "sandbox_service_endpoint", fake_endpoint)
    monkeypatch.setattr(creative_writings, "sandbox_solver", lambda *args: fake_solve_episode)
    monkeypatch.setattr(
        creative_writings,
        "get_model",
        lambda **kwargs: get_model("mockllm/model", custom_outputs=failing_model, memoize=False),
    )

    with pytest.raises(RuntimeError, match="Reviewer failed: RuntimeError: upstream 503"):
        await creative_writings.creative_writings_solver()(state, AsyncMock())

    assert state.metadata["review_errors"] == ["RuntimeError: upstream 503"]


@pytest.mark.parametrize("episode_fails", [False, True], ids=["success", "solver-error"])
async def test_solver_normal_lifecycle_and_error_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, episode_fails: bool
) -> None:
    state = _state(tmp_path)
    box = _fake_sandbox()
    events: list[str] = []
    proxy_closed = anyio.Event()

    async def fake_sandbox_service(
        name: str,
        methods: list[Callable[..., Coroutine[Any, Any, Any]]],
        until: Callable[[], bool],
        environment: Any,
        *,
        user: str,
        started: anyio.Event,
    ) -> None:
        events.append("service:start")
        started.set()
        try:
            await proxy_closed.wait()
            assert until()
        finally:
            events.append("service:stop")

    @asynccontextmanager
    async def fake_endpoint(config: dict[str, Any], *, user: str) -> AsyncGenerator[str, None]:
        events.append("proxy:open")
        try:
            yield "http://127.0.0.1:41234/review"
        finally:
            events.append("proxy:close")
            proxy_closed.set()

    async def fake_solve_episode(st: TaskState, gen: Generate) -> TaskState:
        events.append("agent:run")
        if episode_fails:
            raise ValueError("harness crash")
        return st

    monkeypatch.setattr(creative_writings, "sandbox", lambda: box)
    monkeypatch.setattr(creative_writings, "sandbox_service", fake_sandbox_service)
    monkeypatch.setattr(creative_writings, "sandbox_service_endpoint", fake_endpoint)
    monkeypatch.setattr(creative_writings, "sandbox_solver", lambda *args: fake_solve_episode)
    monkeypatch.setattr(
        creative_writings,
        "get_model",
        lambda **kwargs: get_model("mockllm/model", memoize=False),
    )

    if episode_fails:
        with pytest.raises(ExceptionGroup) as exc_info:
            await creative_writings.creative_writings_solver()(state, AsyncMock())
        assert [type(exc) for exc in exc_info.value.exceptions] == [ValueError]
    else:
        assert await creative_writings.creative_writings_solver()(state, AsyncMock()) is state
        review_cfg = json.loads(str(box.written[creative_writings.REVIEW_FILE]))
        assert review_cfg["url"] == "http://127.0.0.1:41234/review"
        assert len(review_cfg["token"]) == 32

    assert "proxy:close" in events
    assert "service:stop" in events
    assert state.metadata["review_errors"] == []


async def test_solver_rejects_unsupported_sandbox_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    box = _fake_sandbox(arch="riscv64")
    monkeypatch.setattr(creative_writings, "sandbox", lambda: box)

    with pytest.raises(RuntimeError, match="riscv64"):
        await creative_writings.creative_writings_solver()(state, AsyncMock())
