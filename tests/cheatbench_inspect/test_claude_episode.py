import io
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

from cheatbench_inspect.resources.claude import episode


def test_options():
    task = {"model": "model", "tools": {}}
    opts = episode.build_options_kwargs(task, {}, ["mcp__verifier__verify_factors"])
    assert opts["tools"] == ["Bash", "Read", "Write"]
    assert opts["allowed_tools"] == [*opts["tools"], "mcp__verifier__verify_factors"]
    assert opts["disallowed_tools"] == [
        "Task",
        "TodoWrite",
        "Edit",
        "NotebookEdit",
        "WebSearch",
        "WebFetch",
    ]
    task["tools"] = {"mcp_only": True}
    opts = episode.build_options_kwargs(task, {}, [])
    assert opts["tools"] == []
    assert opts["disallowed_tools"] == [
        "Task",
        "TodoWrite",
        "Edit",
        "NotebookEdit",
        "WebSearch",
        "WebFetch",
        "Bash",
        "Read",
        "Write",
        "Agent",
        "Glob",
        "Grep",
        "Skill",
        "ToolSearch",
    ]
    assert opts["strict_mcp_config"]
    assert opts["include_partial_messages"]
    assert opts["setting_sources"] == []


def test_transcript_helpers():
    image = [{"type": "image", "source": {"data": "abc"}}]
    assert (
        episode._elide_images(image)[0]["text"]
        == "<image returned to the model, 3 b64 chars, elided>"
    )
    rec = {
        "_type": "AssistantMessage",
        "content": [
            {"_type": "TextBlock", "text": "hello"},
            {"_type": "ToolUseBlock", "name": "Read"},
        ],
    }
    slim = episode._slim(rec)
    assert slim is not None
    assert episode._count_tool_uses([slim]) == 1
    assert episode._final_text([slim], None) == "hello"
    assert episode._slim({"_type": "StreamEvent"}) is None


def test_main_installer(capsys):
    with patch("sys.stdin", io.StringIO(json.dumps({"task": {"install": "yes"}}))):
        episode.main()
    assert "installers are not supported" in json.loads(capsys.readouterr().out)["error"]


def test_main_memory_modules(tmp_path, capsys):
    root = Path(episode.__file__).parent.parent / "prime_factorization"
    modules = {
        name: (root / filename).read_text()
        for name, filename in [
            ("verifier", "verifier.py"),
            ("grader", "sdk_verifier.py"),
            ("observer", "observer.py"),
        ]
    }
    sdk = types.ModuleType("claude_agent_sdk")
    sdk.__dict__["tool"] = lambda *args: lambda fn: fn
    sdk.__dict__["create_sdk_mcp_server"] = lambda **kwargs: kwargs
    task = {
        "id": "x",
        "model": "m",
        "row": {
            "id": "x",
            "number": "15",
            "live_log_dir": str(tmp_path / "logs"),
            "cache_dir": str(tmp_path / "cache"),
        },
    }

    async def fake_run(task, grader, **kwargs):
        assert grader[1] == ["mcp__verifier__verify_factors"]
        assert kwargs["observer"] is not None
        return {"error": None, "grader_state": grader[2]}

    with (
        patch.dict(sys.modules, {"claude_agent_sdk": sdk}),
        patch("sys.stdin", io.StringIO(json.dumps({"task": task, "modules": modules}))),
        patch.object(episode, "run", fake_run),
    ):
        episode.main()
    assert json.loads(capsys.readouterr().out)["error"] is None
