"""Compare the stock Codex episode against the pre-switch wire and judge fixtures."""

import json
from pathlib import Path

import pytest
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox
from tests.cheatbench_inspect.codex_test_support import (
    normalize,
    response_message,
    run_q4,
    scripted_turns,
)

pytestmark = [pytest.mark.docker, pytest.mark.slow, pytest.mark.dataset_download]

_FIXTURES = Path(__file__).parent / "fixtures/codex_parity"


def test_first_request_and_judge_prompt(tmp_path: Path) -> None:
    log, requests, prompts = run_q4(tmp_path)
    assert log.status == "success", log.error
    assert len(requests) == 3
    legacy = json.loads((_FIXTURES / "first_request.json").read_text(encoding="utf-8"))
    first = json.loads(normalize(json.dumps(requests[0], ensure_ascii=False)))
    assert first == legacy

    [judge_prompt] = prompts
    assert normalize(judge_prompt) == (_FIXTURES / "judge_prompt.txt").read_text(encoding="utf-8")


def test_stock_codex_replays_reasoning_and_text(tmp_path: Path) -> None:
    turns = scripted_turns()
    turns[0].insert(1, response_message("msg_q4_progress", "Submitting the factors.", "commentary"))
    observed: list[str] = []

    @solver
    def capture_rollout() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            result = await sandbox().exec(
                [
                    "find",
                    "/tmp/codex-home/sessions",
                    "-name",
                    "rollout-*.jsonl",
                    "-exec",
                    "cat",
                    "{}",
                    "+",
                ],
                user="user",
            )
            assert result.success, result.stderr
            observed.append(result.stdout)
            return state

        return solve

    log, requests, prompts = run_q4(tmp_path, turns=turns, after=capture_rollout())
    assert log.status == "success", log.error
    assert len(requests) == 3
    replay = [
        {
            "reasoning": [item for item in request["input"] if item["type"] == "reasoning"],
            "assistant": [item for item in request["input"] if item.get("role") == "assistant"],
        }
        for request in requests[1:]
    ]
    assert replay == json.loads((_FIXTURES / "reasoning_replay.json").read_text(encoding="utf-8"))
    assert not any(
        tag in json.dumps(request["input"])
        for tag in ("<think", "<content-internal")
        for request in requests[1:]
    )

    [rollout] = observed
    assistant_text = [
        part["text"]
        for line in rollout.splitlines()
        if (entry := json.loads(line))["type"] == "response_item"
        and (item := entry["payload"])["type"] == "message"
        and item.get("role") == "assistant"
        for part in item["content"]
    ]
    assert assistant_text == json.loads(
        (_FIXTURES / "cli_assistant_text.json").read_text(encoding="utf-8")
    )
    [judge_prompt] = prompts
    for reasoning, *_ in turns:
        assert f"THINKING: {reasoning['summary'][0]['text']}" in judge_prompt
    assert "<think" not in judge_prompt
    assert "<content-internal" not in judge_prompt
