# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/terminus_2_bridge.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

"""Run inside Harbor's isolated Python environment in the episode container."""

# Harbor is installed only in the Docker target.
# pyright: reportMissingImports=false

import asyncio
import contextlib
import json
import os
import shutil
import signal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harbor.models.trial.paths import EnvironmentPaths


class EpisodeEnvironment:
    """Harbor's environment operations within the already isolated container."""

    default_user = None

    def __init__(self, workdir, logs_dir, session_id, env: dict[str, str]):
        self.workdir = workdir
        self.trial_paths = SimpleNamespace(agent_dir=logs_dir)
        self.session_id = session_id
        self.env = env

    async def exec(
        self, command, *, cwd=None, env: dict[str, str] | None = None, timeout_sec=60, user=None
    ):
        # Harbor runs under the same container user for the whole episode.
        if user not in (None, "root", 0, os.getuid()):
            raise ValueError(f"unsupported episode user: {user}")
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd or self.workdir,
            env={**self.env, **(env or {})},
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            raise
        return ExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )

    async def is_dir(self, path):
        return await asyncio.to_thread(Path(path).is_dir)

    async def upload_file(self, source_path, target_path):
        await asyncio.to_thread(self._copy_file, source_path, target_path)

    @staticmethod
    def _copy_file(source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(source_path).resolve() != Path(target_path).resolve():
            shutil.copyfile(source_path, target_path)

    async def download_file(self, source_path, target_path):
        await self.upload_file(source_path, target_path)


def read_artifacts(logs):
    return {p.name: json.loads(p.read_text()) for p in logs.glob("trajectory*.json")}


def observed(agent_class):
    """Mirror each model turn and terminal batch to the task's live log observer."""

    # Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/terminus_2_bridge.py#L82-L130
    class ObservedTerminus2(agent_class):
        observer = None

        async def _handle_llm_interaction(self, *args, **kwargs):
            result = await super()._handle_llm_interaction(*args, **kwargs)
            *_, response = result
            if self.observer:
                blocks = []
                if response.reasoning_content:
                    blocks.append(
                        {"_type": "ThinkingBlock", "thinking": response.reasoning_content}
                    )
                blocks.append({"_type": "TextBlock", "text": response.content})
                self.observer.observe({"_type": "AssistantMessage", "content": blocks})
            return result

        async def _execute_commands(self, commands, session):
            call_id = f"terminal-batch-{self._n_episodes}"
            if self.observer and commands:
                batch = [{"keystrokes": c.keystrokes, "duration": c.duration_sec} for c in commands]
                self.observer.observe(
                    {
                        "_type": "AssistantMessage",
                        "content": [
                            {
                                "_type": "ToolUseBlock",
                                "id": call_id,
                                "name": "Bash",
                                "input": {"commands": batch},
                            }
                        ],
                    }
                )
            result = await super()._execute_commands(commands, session)
            if self.observer:
                self.observer.observe(
                    {
                        "_type": "UserMessage",
                        "content": [
                            {
                                "_type": "ToolResultBlock",
                                "tool_use_id": call_id,
                                "content": result[1],
                            }
                        ],
                    }
                )
            return result

    return ObservedTerminus2


async def episode(config, agent_class, observer_class=None):
    context = AgentContext()
    observer = None
    if config.get("observer"):
        if observer_class is None:
            raise ValueError("an observer configuration needs an observer class")
        observer = observer_class(config["observer"]["row"], config["observer"]["staged"])
        agent_class = observed(agent_class)
    await asyncio.to_thread(Path(EnvironmentPaths.agent_dir).mkdir, parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="terminal-session-") as directory:
        logs = Path(directory)
        environment = EpisodeEnvironment("/workspace", logs, config["id"], dict(os.environ))
        agent = agent_class(
            logs_dir=logs,
            model_name=config["model"],
            api_base=config["api_base"],
            llm_kwargs={"api_key": "inspect", "timeout": 930},
            mcp_servers=[MCPServerConfig(**server) for server in config["mcp_servers"]],
            max_turns=config["max_turns"],
        )
        agent.options.llm_kwargs = {}
        agent.observer = observer
        error = None
        try:
            await asyncio.wait_for(agent.setup(environment), 600)
            await asyncio.wait_for(agent.run(config["prompt"], environment, context), 3600)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if agent._session is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(agent._session.stop(), 15)

        live_logs = None
        if observer:
            try:
                live_logs = observer.finish()
            except Exception as exc:
                error = f"{error or ''} observer finish failed: {exc}".strip()
        artifacts = await asyncio.to_thread(read_artifacts, logs)
        trajectory = artifacts.pop("trajectory.json", None)
        if trajectory is None and error is None:
            error = "Harbor produced no trajectory"
        steps = [step for step in (trajectory or {}).get("steps", []) if step["source"] == "agent"]
        limit = config["max_turns"]
        exhausted = limit and (context.metadata or {}).get("n_episodes", 0) >= limit
        if (
            not error
            and exhausted
            and (
                len(steps) < 2
                or not all(
                    any(
                        call["function_name"] == "mark_task_complete"
                        for call in step.get("tool_calls") or []
                    )
                    for step in steps[-2:]
                )
            )
        ):
            error = "max_turns reached before confirmed task completion"
        return {
            "final_text": (steps[-1].get("message") or "") if steps else "",
            "harbor_trajectory": trajectory,
            "harbor_artifacts": artifacts,
            "live_logs": live_logs,
            "error": error,
        }
