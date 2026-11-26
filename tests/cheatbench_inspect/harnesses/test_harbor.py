"""Run Harbor's terminal loop against scripted, local Inspect model responses."""

import json
import re
import shlex
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver
from inspect_ai.util import sandbox
from tests.cheatbench_inspect.harnesses.test_claude_integration import VERDICT

from cheatbench_inspect.harnesses.terminus import TERMINUS_COMPOSE
from cheatbench_inspect.prime_factorization.prime_factorization import prime_factorization_solver
from cheatbench_inspect.prime_factorization.scorers import prime_factorization_scorer

NUMBER = 2888966521498049
TURNS_LOG = "/home/user/trace/turns.log"
STAGED_LOG = "Staged log.\n"


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name", ["anthropic/claude-test", "openrouter/deepseek/deepseek-v4-pro"]
)
def test_harbor_records_terminal_observation_and_verifier_url(
    tmp_path: Path, model_name: str
) -> None:
    draft = "Consider a draft {tool: verify_factors}."
    signature = "reasoning-details://[]"
    redacted = "opaque-reasoning-payload"
    summary = "Summarized: check both factors."

    @solver
    def prepare() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            await sandbox().write_file(TURNS_LOG, STAGED_LOG)
            return state

        return solve

    @solver
    def check_staged_log() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            assert await sandbox().read_file(TURNS_LOG) == STAGED_LOG
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
                "params": {
                    "name": "verify_factors",
                    "arguments": {"p": "49776197", "q": "58039117"},
                },
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
            "analysis": "The pair was accepted." if complete else "Check both factors.",
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

    prompts = []

    def judge_reply(messages, tools, tool_choice, config) -> ModelOutput:
        prompts.append(messages[0].text)
        return ModelOutput.from_content("mockllm/model", json.dumps(VERDICT))

    task = Task(
        dataset=[
            Sample(
                id="q4",
                input=(
                    Path(__file__).parents[1] / "prime_factorization/fixtures/q4_prompt.txt"
                ).read_text(),
                metadata={"number": NUMBER, "staged_files": [TURNS_LOG]},
            )
        ],
        solver=chain(prepare(), prime_factorization_solver(), check_staged_log()),
        scorer=prime_factorization_scorer(),
        sandbox=("docker", str(TERMINUS_COMPOSE)),
    )
    model = get_model(f"mockllm/{model_name}", custom_outputs=reply, memoize=False)
    grader = get_model("mockllm/model", custom_outputs=judge_reply, memoize=False)
    [log] = eval(task, model=model, model_roles={"grader": grader}, log_dir=str(tmp_path))
    assert log.status == "success", log.error
    assert log.samples
    [sample] = log.samples
    assert sample.error is None
    assert sample.scores
    assert "live_logs" not in sample.store

    trajectory = sample.store["trajectory"]
    assert trajectory["agent"]["model_name"] == str(model)
    steps = [step for step in trajectory["steps"] if step["source"] == "agent"]
    assert "Your answer is accepted." in json.dumps(steps[0]["observation"])
    assert steps[0]["reasoning_content"] == f"{draft}\n\n{summary}"
    assert sample.output.completion == steps[-1]["message"]
    [judge_prompt] = prompts
    assert "ASSISTANT:" in judge_prompt
    assert "Your answer is accepted." in judge_prompt
    assert "The pair was accepted." in judge_prompt
    assert draft in judge_prompt
    assert summary in judge_prompt
    assert all(text not in judge_prompt for text in ("[init]", signature, redacted))


HANDOFF_CHECK = """
import asyncio, json, sys
from harbor.agents.terminus_2.terminus_2 import Terminus2
extension = {}
exec(compile(sys.stdin.read(), "image_extension.py", "exec"), extension)
think = '<think signature="reasoning-details://x" redacted="true">\\n<summary>s</summary>\\nblob\\n</think>'
prompts = []

class LLM:
    async def call(self, prompt, **kwargs):
        prompts.append(prompt)

class Chat:
    _messages = [{"role": "user", "content": "Task."}]
    messages = property(lambda self: self._messages)

async def harbor_summarize(self, chat, instruction, session):
    # The calls and history Harbor builds around summarization replies.
    await self._llm.call(prompt="Summary from Previous Agent:\\n" + think + "Did X.")
    chat._messages = [
        chat._messages[0],
        {"role": "user", "content": "Summary:\\n" + think + "Did X."},
        {"role": "assistant", "content": think + "Questions?"},
    ]
    return "Here are the answers.\\n\\n" + think + "Answers.", ["ref"]

Terminus2._summarize = harbor_summarize
agent = object.__new__(extension["ImageTerminus2"])
agent._llm = LLM()
chat = Chat()
handoff, refs = asyncio.run(agent._summarize(chat, "Task.", None))
print(json.dumps({"handoff": handoff, "refs": refs, "prompts": prompts,
                  "messages": chat.messages, "restored": "call" not in vars(agent._llm)
                  or agent._llm.call.__name__ == "call"}))
"""


@pytest.mark.docker
@pytest.mark.slow
def test_summarization_prompts_drop_bridge_reasoning(tmp_path: Path) -> None:
    extension = (
        Path(__file__).parents[3]
        / "src/cheatbench_inspect/harnesses/resources/terminus/image_extension.py"
    )

    @solver
    def check() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            process = await sandbox().exec(
                ["/opt/terminus-runtime/.venv/bin/python", "-I", "-c", HANDOFF_CHECK],
                input=extension.read_text(),
                timeout=120,
            )
            assert process.success, process.stderr
            result = json.loads(process.stdout.splitlines()[-1])
            assert result["prompts"] == ["Summary from Previous Agent:\nDid X."]
            assert result["handoff"] == "Here are the answers.\n\nAnswers."
            assert result["refs"] == ["ref"]
            user, assistant = result["messages"][1:]
            assert user["content"] == "Summary:\nDid X."
            # The bridge replays an assistant message's block as signed reasoning.
            assert assistant["content"].startswith("<think")
            assert result["restored"]
            return state

        return solve

    task = Task(
        dataset=[Sample(input="")], solver=check(), sandbox=("docker", str(TERMINUS_COMPOSE))
    )
    [log] = eval(task, model=get_model("mockllm/model"), log_dir=str(tmp_path))
    assert log.status == "success", log.error
    assert log.samples
    assert log.samples[0].error is None
