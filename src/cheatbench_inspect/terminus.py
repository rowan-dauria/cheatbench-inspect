"""Prepare CheatBench's Terminus-2 harness inside the active Inspect sandbox."""

import json
from pathlib import Path
from typing import Any

from cheatbench_inspect.container import BOOTSTRAP

_RESOURCES = Path(__file__).parent / "resources/terminus"
TERMINUS_COMPOSE = _RESOURCES.parent / "sandbox/terminus-compose.yaml"

# The episode receives trusted code on stdin. Harbor may print diagnostics during
# imports and inference, so reserve stdout for the completed result.
_START = """
import asyncio, contextlib, json, os, sys
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
payload = PAYLOAD
with contextlib.redirect_stdout(sys.stderr):
    extension, runner = {}, {}
    exec(compile(payload["extension"], "image_extension.py", "exec"), extension)
    exec(compile(payload["runner"], "episode.py", "exec"), runner)
    observer = None
    if payload["observer"]:
        task = {}
        exec(compile(payload["observer"], "observer.py", "exec"), task)
        observer = task["MessageObserver"]
    result = asyncio.run(
        runner["episode"](payload["config"], extension["ImageTerminus2"], observer)
    )
print(json.dumps(result))
"""


def prepare_harbor(
    *,
    sample_id: str,
    prompt: str,
    model: str,
    api_base: str,
    mcp_servers: list[dict[str, str]],
    max_turns: int | None = None,
    observer: dict[str, Any] | None = None,
) -> tuple[list[str], str]:
    """Return the sandbox command and the JSON document to send on its stdin.

    `observer` carries the task's live log observer: its `code`, which defines
    `MessageObserver`, and the `row` and `staged` paths it is constructed with.
    """
    payload = {
        "code": _START,
        "config": {
            "id": sample_id,
            "prompt": prompt,
            "model": model,
            "api_base": api_base,
            "mcp_servers": mcp_servers,
            "max_turns": max_turns,
            "observer": {k: v for k, v in observer.items() if k != "code"} if observer else None,
        },
        "extension": (_RESOURCES / "image_extension.py").read_text(),
        "runner": (_RESOURCES / "episode.py").read_text(),
        "observer": observer["code"] if observer else None,
    }
    return ["/opt/terminus-runtime/.venv/bin/python", "-I", "-c", BOOTSTRAP], json.dumps(payload)
