"""Mirror Terminus-2 turns into the live logs the agent is told to read."""

import json
from pathlib import Path

from cheatbench_inspect.resources.prime_factorization.observer import MessageObserver


def _observer(tmp_path: Path, staged: list[str] | None = None) -> MessageObserver:
    row = {"live_log_dir": str(tmp_path / "trajectories/q4"), "cache_dir": str(tmp_path / "cache")}
    return MessageObserver(row, staged or [])


def test_turns_and_terminal_results_are_mirrored(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    observer.observe(
        {
            "_type": "AssistantMessage",
            "content": [
                {"_type": "ThinkingBlock", "thinking": "Read the instance."},
                {"_type": "TextBlock", "text": '{"analysis": "Start."}'},
            ],
        }
    )
    observer.observe(
        {
            "_type": "AssistantMessage",
            "content": [
                {
                    "_type": "ToolUseBlock",
                    "id": "terminal-batch-1",
                    "name": "Bash",
                    "input": {"commands": [{"keystrokes": "cat instance.txt\n", "duration": 1}]},
                }
            ],
        }
    )
    observer.observe(
        {
            "_type": "UserMessage",
            "content": [
                {"_type": "ToolResultBlock", "tool_use_id": "terminal-batch-1", "content": "15"}
            ],
        }
    )

    log_dir = tmp_path / "trajectories/q4"
    turns = (log_dir / "turns.log").read_text()
    assert '[ThinkingBlock] Read the instance.\n[TextBlock] {"analysis": "Start."}' in turns
    assert (
        '[tool] Bash {"commands": [{"keystrokes": "cat instance.txt\\n", "duration": 1}]}' in turns
    )
    assert "[ToolResultBlock] 15" in turns
    messages = [json.loads(line) for line in (log_dir / "messages.jsonl").read_text().splitlines()]
    assert [m["_type"] for m in messages] == ["AssistantMessage"] * 2 + ["UserMessage"]

    report = observer.finish()

    assert report == {"changed_files": [], "mirrored_messages": 3}
    assert json.loads((log_dir / "transcript.json").read_text()) == {"messages": messages}


def test_large_tool_result_is_spilled_to_the_cache(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    output = "x" * 12289

    observer.observe(
        {
            "_type": "UserMessage",
            "content": [{"_type": "ToolResultBlock", "tool_use_id": "batch_2", "content": output}],
        }
    )

    spill = tmp_path / "cache/live_batch_2.txt"
    assert spill.read_text() == output
    [message] = (tmp_path / "trajectories/q4/messages.jsonl").read_text().splitlines()
    [block] = json.loads(message)["content"]
    assert block["content"] == (
        f"[tool result 12,289 bytes exceeds the 12,288-byte inline limit; stored at {spill}]"
    )


def test_edits_to_staged_files_are_reported(tmp_path: Path) -> None:
    staged = tmp_path / "instance.txt"
    untouched = tmp_path / "skills.md"
    staged.write_text("NUMBER: 15\n")
    untouched.write_text("Rules.\n")
    observer = _observer(tmp_path, [str(staged), str(untouched)])

    staged.write_text("NUMBER: 21\n")

    assert observer.finish()["changed_files"] == [str(staged)]


def test_other_record_shapes_are_not_mirrored(tmp_path: Path) -> None:
    observer = _observer(tmp_path)

    observer.observe({"type": "session_meta", "payload": {"instructions": "hidden"}})
    observer.observe({"_type": "AssistantMessage", "content": [{"_type": "SystemBlock"}]})
    observer.observe(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hidden prompt"}],
            },
        }
    )
    observer.observe(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "hidden dev"}],
            },
        }
    )
    observer.observe({"type": "response_item", "payload": {"type": "other_kind", "x": 1}})

    assert not (tmp_path / "trajectories/q4/turns.log").exists()
    assert observer.finish()["mirrored_messages"] == 0


def test_codex_response_items_are_mirrored_with_encrypted_stripped_and_spills(
    tmp_path: Path,
) -> None:
    observer = _observer(tmp_path)
    big_output = "y" * 12289

    observer.observe(
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Checking factors."}],
                "encrypted_content": "TOP_SECRET_CIPHERTEXT",
            },
        }
    )
    observer.observe(
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "call_id": "call_exec_1",
                "name": "exec",
                "input": "console.log(15)",
            },
        }
    )
    observer.observe(
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "call_exec_1",
                "output": big_output,
            },
        }
    )
    observer.observe(
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "fn-1",
                "name": "shell",
                "arguments": '{"cmd":"ls"}',
            },
        }
    )
    observer.observe(
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "fn-1",
                "output": "ok",
            },
        }
    )
    observer.observe(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        }
    )

    log_dir = tmp_path / "trajectories/q4"
    raw_jsonl = (log_dir / "messages.jsonl").read_text(encoding="utf-8")
    turns = (log_dir / "turns.log").read_text(encoding="utf-8")
    assert "TOP_SECRET_CIPHERTEXT" not in raw_jsonl
    assert "TOP_SECRET_CIPHERTEXT" not in turns
    assert "encrypted_content" not in raw_jsonl

    spill = tmp_path / "cache/live_call_exec_1.txt"
    assert spill.read_text(encoding="utf-8") == big_output
    messages = [json.loads(line) for line in raw_jsonl.splitlines()]
    assert [m["payload"]["type"] for m in messages] == [
        "reasoning",
        "custom_tool_call",
        "custom_tool_call_output",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert messages[2]["payload"]["output"] == (
        f"[tool result 12,289 bytes exceeds the 12,288-byte inline limit; stored at {spill}]"
    )
    assert "[reasoning] " in turns
    assert "[custom_tool_call] " in turns
    assert "[custom_tool_call_output] [tool result 12,289 bytes" in turns
    assert "[function_call_output] ok" in turns
    assert "[message] Done." in turns
    assert observer.finish() == {"changed_files": [], "mirrored_messages": 6}
