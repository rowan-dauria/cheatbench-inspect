"""Stream bridge messages to the task's in-sandbox live log observer."""

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.util import ExecRemoteStreamingOptions, ExecStderr, ExecStdout, sandbox

from cheatbench_inspect.container import AGENT_ENV, AGENT_USER, STREAM_BOOTSTRAP
from cheatbench_inspect.harness import HarnessPolicy

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


@asynccontextmanager
async def live_log(
    policy: HarnessPolicy,
    report: dict[str, Any],
    records: Callable[[ChatMessageAssistant | ChatMessageTool], list[dict[str, Any]]],
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
                await process.write_stdin(
                    json.dumps([record for message in messages for record in records(message)])
                    + "\n"
                )
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
