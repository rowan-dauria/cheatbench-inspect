"""Run inspect_swe's Claude Code agent and capture its judge trajectory."""

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from inspect_ai.agent import AgentState
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    GenerateConfig,
    Model,
    ModelOutput,
)
from inspect_ai.solver import TaskState
from inspect_ai.tool import MCPServerConfigHTTP, ToolChoice, ToolInfo
from inspect_ai.util import ExecRemoteStreamingOptions, ExecStderr, ExecStdout, sandbox
from inspect_swe import claude_code

from cheatbench_inspect.agent_trajectory import message_text, readable_reasoning, to_trajectory
from cheatbench_inspect.container import AGENT_ENV, AGENT_USER, STREAM_BOOTSTRAP
from cheatbench_inspect.harness import HarnessPolicy, HarnessResult
from cheatbench_inspect.lost_episodes import MAX_TURNS_EXACT_ERROR, is_harness_timeout
from cheatbench_inspect.reasoning_replay import anthropic_reasoning_filter

_DISALLOWED_TOOLS = [
    "Agent",
    "Task",
    "TaskOutput",
    "TaskStop",
    "Monitor",
    "TodoWrite",
    "Edit",
    "NotebookEdit",
    "WebSearch",
    "WebFetch",
    "Glob",
    "Grep",
    "Skill",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "EnterWorktree",
    "ExitWorktree",
    "ListMcpResourcesTool",
    "ReadMcpResourceTool",
    "ToolSearch",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
    "CronCreate",
    "CronDelete",
    "CronList",
    "ListAgents",
    "ReportFindings",
    "ScheduleWakeup",
    "SendMessage",
    "Workflow",
]


_START_OBSERVER = """
import json, sys
observer = {}
exec(compile(PAYLOAD["observer"], "observer.py", "exec"), observer)
mirror = observer["MessageObserver"](PAYLOAD["row"], PAYLOAD["staged"])
for line in sys.stdin:
    for message in json.loads(line):
        mirror.observe(message)
    print(flush=True)
print(json.dumps(mirror.finish()), flush=True)
"""


def _mirror_record(message: ChatMessageAssistant | ChatMessageTool) -> dict[str, Any]:
    """Adapt a bridge message to the original live log observer's record format."""
    text = message_text(message)
    if isinstance(message, ChatMessageTool):
        return {
            "_type": "UserMessage",
            "content": [
                {
                    "_type": "ToolResultBlock",
                    "tool_use_id": message.tool_call_id,
                    "content": text or (message.error.message if message.error else ""),
                    "is_error": message.error is not None,
                }
            ],
        }
    blocks: list[dict[str, Any]] = []
    if thinking := readable_reasoning(message):
        blocks.append({"_type": "ThinkingBlock", "thinking": thinking})
    if text:
        blocks.append({"_type": "TextBlock", "text": text})
    blocks.extend(
        {"_type": "ToolUseBlock", "id": call.id, "name": call.function, "input": call.arguments}
        for call in message.tool_calls or []
    )
    return {"_type": "AssistantMessage", "content": blocks}


@asynccontextmanager
async def _live_log(
    policy: HarnessPolicy, report: dict[str, Any]
) -> AsyncGenerator[Callable[[list[ChatMessageAssistant | ChatMessageTool]], Awaitable[None]]]:
    process = None
    if observer := policy.observer:
        payload = {**observer, "observer": observer["code"], "code": _START_OBSERVER}
        process = await sandbox().exec_remote(
            ["python3", "-I", "-c", STREAM_BOOTSTRAP],
            ExecRemoteStreamingOptions(
                input=json.dumps(payload) + "\n", stdin_open=True, user=AGENT_USER, env=AGENT_ENV
            ),
        )

    if process is None:

        async def ignore(messages: list[ChatMessageAssistant | ChatMessageTool]) -> None:
            pass

        yield ignore
        return

    events = aiter(process)
    write_lock = asyncio.Lock()
    stdout = ""
    stderr = ""

    async def read_line() -> str:
        nonlocal stdout, stderr
        async with asyncio.timeout(30):
            while "\n" not in stdout:
                event = await anext(events, None)
                if event is None:
                    raise RuntimeError(f"Live log observer exited: {stderr.strip()}")
                if isinstance(event, ExecStdout):
                    stdout += event.data
                elif isinstance(event, ExecStderr):
                    stderr += event.data
        line, stdout = stdout.split("\n", 1)
        return line

    async def write(messages: list[ChatMessageAssistant | ChatMessageTool]) -> None:
        if messages:
            async with write_lock:
                await process.write_stdin(json.dumps([_mirror_record(m) for m in messages]) + "\n")
                # The next tool may read turns.log. Wait until the writes are visible.
                await read_line()

    try:
        yield write
        await process.close_stdin()
        report.update(json.loads(await read_line()))
        while await anext(events, None) is not None:
            pass
    finally:
        await asyncio.shield(process.kill())


class ClaudeTurnLimitError(Exception):
    """The CLI requested another model call after the episode's turn budget."""


async def run_claude_code(
    state: TaskState, policy: HarnessPolicy, *, cli_version: str, cli_model: str
) -> HarnessResult:
    """Run one pinned Claude Code episode inside the sample sandbox."""
    report: dict[str, Any] = {}
    async with _live_log(policy, report) as mirror:
        result = await _run_claude_code(state, policy, cli_version, cli_model, mirror)
    result.live_logs = report or None
    return result


async def _run_claude_code(
    state: TaskState,
    policy: HarnessPolicy,
    cli_version: str,
    cli_model: str,
    mirror: Callable[[list[ChatMessageAssistant | ChatMessageTool]], Awaitable[None]],
) -> HarnessResult:
    turns = 0
    first_tools: list[str] = []
    observed: list[ChatMessage] = []
    seen_tool_results: set[str | None] = set()
    bridge_error: str | None = None
    turn_limit = asyncio.Event()

    async def filter_request(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput:
        nonlocal turns, first_tools, bridge_error
        tool_results: list[ChatMessageAssistant | ChatMessageTool] = []
        for message in messages:
            if (
                isinstance(message, ChatMessageTool)
                and message.tool_call_id not in seen_tool_results
            ):
                seen_tool_results.add(message.tool_call_id)
                tool_results.append(message)
        observed.extend(tool_results)
        await mirror(tool_results)
        if policy.max_turns is not None and turns >= policy.max_turns:
            turn_limit.set()
            raise ClaudeTurnLimitError(MAX_TURNS_EXACT_ERROR)
        turns += 1
        if not first_tools:
            first_tools = [tool.name for tool in tools]
        try:
            output = await anthropic_reasoning_filter(model, messages, tools, tool_choice, config)
        except Exception as exc:
            bridge_error = f"{type(exc).__name__}: {exc}"
            raise
        bridge_error = None
        if not output.empty:
            observed.append(output.message)
            await mirror([output.message])
        return output

    agent = claude_code(
        version=cli_version,
        model_config=cli_model,
        cwd="/workspace",
        user=AGENT_USER,
        env={
            **AGENT_ENV,
            "DISABLE_AUTOUPDATER": "1",
            # Keep the SDK request metadata and suppress the extra CLI budget reminder.
            "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
            "CLAUDE_CODE_TOTAL_TOKENS_REMINDER": "off",
        },
        permission_mode="bypassPermissions",
        retry_refusals=0,
        retry_uncaught_errors=0,
        filter=filter_request,
        disallowed_tools=_DISALLOWED_TOOLS,
        replace_system_prompt="",
        mcp_servers=[
            MCPServerConfigHTTP(name=server["name"], url=server["url"], type="http")
            for server in policy.mcp_servers
        ],
    )
    agent_state = AgentState(messages=state.messages.copy())
    execution = asyncio.create_task(agent(agent_state))
    capped = asyncio.create_task(turn_limit.wait())
    try:
        async with asyncio.timeout(policy.timeout_s):
            done, _ = await asyncio.wait((execution, capped), return_when=asyncio.FIRST_COMPLETED)
            if capped in done:
                # The bridge turns filter exceptions into HTTP errors. Stop the CLI
                # ourselves so it cannot keep retrying after the budget is exhausted.
                raise ClaudeTurnLimitError(MAX_TURNS_EXACT_ERROR)
            agent_state = await execution
        error = bridge_error
    except ClaudeTurnLimitError as exc:
        error = str(exc)
    except TimeoutError:
        error = f"timeout after {policy.timeout_s}s"
    except Exception as exc:
        error = "\n".join(dict.fromkeys(filter(None, (str(exc).strip(), bridge_error))))
    finally:
        execution.cancel()
        capped.cancel()
        await asyncio.gather(execution, capped, return_exceptions=True)

    # Bridge state is the latest conversation. Keep every observed turn if the
    # CLI compacted that history or a failed request brought the last results.
    messages = agent_state.messages
    if error or sum(isinstance(m, ChatMessageAssistant) for m in messages) < sum(
        isinstance(m, ChatMessageAssistant) for m in observed
    ):
        messages = observed
    output = agent_state.output
    if error is None and not output.empty and output.stop_reason == "content_filter":
        error = "model_refusal_no_fallback"
    final_text = (
        output.completion.strip() if not output.empty and output.stop_reason != "tool_calls" else ""
    )
    details = {
        "tools": first_tools,
        "mcp_servers": [
            {"name": server["name"], "status": "connected"}
            for server in policy.mcp_servers
            if any(name.startswith(f"mcp__{server['name']}__") for name in first_tools)
        ],
    }
    trajectory = to_trajectory(
        messages,
        agent="claude-code",
        version=cli_version,
        model=cli_model,
        details=details,
        final_text=final_text,
    )
    if trajectory is None and error is None:
        error = "Claude Code produced no model response"
    return HarnessResult(
        trajectory, final_text, error, {}, None, lost=False if is_harness_timeout(error) else None
    )
