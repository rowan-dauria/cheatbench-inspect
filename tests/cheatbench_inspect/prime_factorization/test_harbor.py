"""Run Harbor's terminal loop against scripted, local Inspect model responses."""

import json
import re
import shlex
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import sandbox_agent_bridge
from inspect_ai.dataset import Sample
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from cheatbench_inspect.container import AGENT_ENV, AGENT_USER, hand_staged_files_to_agent
from cheatbench_inspect.judge import render_trajectory
from cheatbench_inspect.prime_factorization import live_log_observer, verifier_endpoint
from cheatbench_inspect.terminus import TERMINUS_COMPOSE, prepare_harbor

REPO = "/home/runner/work/factoring-evals/factoring-evals"
LIVE_LOG_DIR = f"{REPO}/logs/trajectories/15"
STAGED = f"{REPO}/README.md"


@pytest.mark.docker
@pytest.mark.slow
def test_harbor_records_terminal_observation_and_verifier_url(tmp_path: Path) -> None:
    draft = "Consider a draft {tool: verify_factors}."
    signature = "reasoning-details://[]"
    redacted = "opaque-reasoning-payload"
    summary = "Summarized: check 3 and 5."

    @solver
    def smoke() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            await sandbox().write_file(STAGED, "Factoring notes.\n")
            await hand_staged_files_to_agent([STAGED])
            async with verifier_endpoint(15) as url, sandbox_agent_bridge():
                command, stdin = prepare_harbor(
                    sample_id="15",
                    prompt=state.input_text,
                    model="openai/gpt-4o",
                    api_base="http://127.0.0.1:13131/v1",
                    mcp_servers=[{"name": "verifier", "transport": "streamable-http", "url": url}],
                    max_turns=5,
                    observer=live_log_observer(
                        "15",
                        {
                            "live_log_dir": LIVE_LOG_DIR,
                            "cache_dir": "/home/runner/work/_temp/tool_results",
                        },
                        [STAGED],
                    ),
                )
                process = await sandbox().exec(
                    command, input=stdin, timeout=120, user=AGENT_USER, env=AGENT_ENV
                )
                assert process.success, process.stderr
                result = json.loads(process.stdout)
            assert result["error"] is None, result["error"]
            live_logs = result["live_logs"]
            assert live_logs["changed_files"] == []
            messages = await sandbox().read_file(f"{LIVE_LOG_DIR}/messages.jsonl")
            assert live_logs["mirrored_messages"] == len(messages.splitlines())
            turns = await sandbox().read_file(f"{LIVE_LOG_DIR}/turns.log")
            assert f"[ThinkingBlock] {draft}" in turns
            assert "[tool] Bash" in turns
            assert "verify_factors" in turns
            assert "Your answer is accepted." in turns
            trajectory = result["harbor_trajectory"]
            steps = trajectory["steps"]
            assert url in json.dumps(steps)
            first_turn = next(step for step in steps if step["source"] == "agent")
            assert "Your answer is accepted." in json.dumps(first_turn.get("observation"))
            assert first_turn.get("reasoning_content") == f"{draft}\n\n{summary}"
            judge_log = render_trajectory(trajectory)
            assert f"THINKING: {draft}" in judge_log
            assert summary in judge_log
            assert signature not in judge_log
            assert redacted not in judge_log
            assert (
                result["final_text"] == [s for s in steps if s["source"] == "agent"][-1]["message"]
            )
            return state

        return solve

    def reply(messages, tools, tool_choice, config) -> ModelOutput:
        complete = any(message.role == "assistant" for message in messages)
        commands = []
        if not complete:
            prompt = "\n".join(message.text for message in messages)
            match = re.search(r"url: (http://127\.0\.0\.1:\d+/[a-f0-9]+/mcp)", prompt)
            assert match, "Harbor must give the verifier URL to the model"
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "verify_factors", "arguments": {"p": "3", "q": "5"}},
            }
            command = shlex.join(
                [
                    "curl",
                    "-fsS",
                    "-H",
                    "Content-Type: application/json",
                    "--data",
                    json.dumps(request),
                    match[1],
                ]
            )
            commands.append({"keystrokes": command + "\n", "duration": 0.5})
        message = {
            "analysis": "The pair was accepted." if complete else "Check 3 and 5.",
            "plan": "Finish." if complete else "Call the verifier.",
            "commands": commands,
            "task_complete": complete,
        }
        if complete:
            assert any(
                isinstance(part, ContentReasoning)
                and part.reasoning == draft
                and part.signature == signature
                for entry in messages
                if entry.role == "assistant" and isinstance(entry.content, list)
                for part in entry.content
            ), "Keep the original signed reasoning in the conversation history"
            return ModelOutput.from_content("mockllm/model", json.dumps(message))

        return ModelOutput.from_content(
            "mockllm/model",
            [
                ContentReasoning(reasoning=draft, signature=signature),
                ContentReasoning(reasoning=redacted, redacted=True, summary=summary),
                ContentText(text=json.dumps(message)),
            ],
        )

    task = Task(
        dataset=[Sample(input="Check 3 times 5.")],
        solver=smoke(),
        sandbox=("docker", str(TERMINUS_COMPOSE)),
    )
    model = get_model("mockllm/model", custom_outputs=reply, memoize=False)
    [log] = eval(task, model=model, log_dir=str(tmp_path))
    assert log.status == "success", log.error
    assert log.samples
    assert log.samples[0].error is None
