import asyncio
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.agent import AgentState
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import solver

from cheatbench_inspect.sandbox_task import sandbox_solver


@pytest.mark.parametrize("harness", ["claude-code", "codex", "gemini-cli"])
def test_timeout_preserves_trajectory_for_scorer(harness: str, tmp_path: Path) -> None:
    seen = []

    async def stage(paths):
        pass

    async def launch(name, state, policy, options):
        state.messages.append(ChatMessageAssistant(content="evidence before timeout"))
        await asyncio.sleep(2)
        return AgentState(messages=state.messages)

    @scorer(metrics=[])
    def judge():
        async def score(state, target):
            seen.extend(message.text for message in state.messages)
            return Score(value=1)

        return score

    @solver
    def episode():
        return sandbox_solver(
            harness, "1", "model", 150, 1, stage_files=stage, launch_harness=launch
        )

    task = Task(
        dataset=[Sample(input="solve", metadata={"staged_files": []})],
        solver=episode(),
        scorer=judge(),
    )
    [log] = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")
    assert log.status == "success"
    assert log.samples
    assert log.samples[0].scores
    assert "evidence before timeout" in seen
    assert any(event.event == "sample_limit" for event in log.samples[0].events)


@pytest.mark.parametrize(
    ("module", "factory", "expected"),
    [
        ("prime_factorization.prime_factorization", "prime_factorization_solver", 3600),
        ("subset_sum.subset_sum", "subset_sum_solver", 3600),
        ("geoguessr.geoguessr", "geoguessr_solver", 3600),
        ("protein_design.protein_design", "protein_design_solver", 5400),
        ("synthetic_puzzles.synthetic_puzzles", "synthetic_puzzles_solver", 1800),
    ],
)
def test_task_passes_episode_timeout(module, factory, expected, monkeypatch):
    from importlib import import_module
    from unittest.mock import Mock

    task_module = import_module(f"cheatbench_inspect.{module}")

    async def fake_solver(state, generate):
        return state

    capture = Mock(return_value=fake_solver)
    monkeypatch.setattr(task_module, "sandbox_solver", capture)
    getattr(task_module, factory)("terminus-2", None, None)
    args = capture.call_args
    assert (args.kwargs["timeout_s"] if "timeout_s" in args.kwargs else args.args[4]) == expected
