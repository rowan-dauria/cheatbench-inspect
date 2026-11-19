"""Prepare CheatBench's Terminus-2 harness inside the active Inspect sandbox."""

import json
from pathlib import Path

_RESOURCES = Path(__file__).parent / "resources/terminus"
TERMINUS_COMPOSE = _RESOURCES.parent / "sandbox/compose-terminus.yaml"

# The episode receives trusted code on stdin. Harbor may print diagnostics during
# imports and inference, so reserve stdout for the completed result.
_START = """
import asyncio, contextlib, json, os, sys
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
payload = json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
    extension, runner = {}, {}
    exec(compile(payload["extension"], "image_extension.py", "exec"), extension)
    exec(compile(payload["runner"], "episode.py", "exec"), runner)
    result = asyncio.run(runner["episode"](payload["config"], extension["ImageTerminus2"]))
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
) -> tuple[list[str], str]:
    """Return the sandbox command and the JSON document to send on its stdin."""
    payload = {
        "config": {
            "id": sample_id,
            "prompt": prompt,
            "model": model,
            "api_base": api_base,
            "mcp_servers": mcp_servers,
            "max_turns": max_turns,
        },
        "extension": (_RESOURCES / "image_extension.py").read_text(),
        "runner": (_RESOURCES / "episode.py").read_text(),
    }
    return ["/opt/terminus-runtime/.venv/bin/python", "-I", "-c", _START], json.dumps(payload)
