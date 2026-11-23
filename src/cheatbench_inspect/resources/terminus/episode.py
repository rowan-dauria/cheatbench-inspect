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


async def episode(config, agent_class):
    context = AgentContext()
    with TemporaryDirectory(prefix="terminal-session-") as directory:
        logs = Path(directory)
        environment = EpisodeEnvironment("/workspace", logs, config["id"], dict(os.environ))
        agent = agent_class(
            logs_dir=logs,
            model_name=config["model"],
            api_base=config["api_base"],
            llm_kwargs={"api_key": "inspect", "timeout": 930, "custom_llm_provider": "openai"},
            mcp_servers=[MCPServerConfig(**server) for server in config["mcp_servers"]],
            max_turns=config["max_turns"],
        )
        agent.options.llm_kwargs = {}
        error = None
        try:
            await asyncio.wait_for(agent.setup(environment), 600)
            await asyncio.wait_for(
                agent.run(config["prompt"], environment, context), config["timeout_s"]
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if agent._session is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(agent._session.stop(), 15)

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
            "error": error,
        }
