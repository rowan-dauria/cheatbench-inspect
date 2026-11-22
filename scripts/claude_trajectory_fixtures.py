"""Generate fixtures using original CheatBench 96e05a6.

From /tmp, run with PYTHONDONTWRITEBYTECODE=1 and uv run python -B in a
throwaway environment containing pydantic, PyYAML and claude-agent-sdk==0.2.135.
Pass the repository path and a captured integration record as arguments.
"""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/agent/cheatbench")
from core.agents.claude_sdk import ClaudeSDKAgent
from core.render import render_trajectory

out = Path(sys.argv[1]) / "tests/cheatbench_inspect/fixtures/claude_sdk"
record = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
envelope = (
    "openrouter.reasoning:"
    + base64.b64encode(
        json.dumps({"text": "private thought", "type": "reasoning.text"}).encode()
    ).decode()
)
records = {
    "integration": record,
    "empty": {},
    "synthetic": {
        "messages": [
            {
                "_type": "SystemMessage",
                "subtype": "init",
                "data": {"tools": ["Bash"], "mcp_servers": []},
            },
            {
                "_type": "AssistantMessage",
                "content": [
                    {"_type": "ThinkingBlock", "thinking": " think "},
                    {"_type": "ThinkingBlock"},
                    {"_type": "RedactedThinkingBlock", "data": "opaque"},
                    {"_type": "RedactedThinkingBlock", "data": envelope},
                    {"_type": "TextBlock", "text": " hello "},
                    {"_type": "ToolUseBlock", "id": "a", "name": "Bash", "input": {"command": "x"}},
                    {"_type": "ToolUseBlock", "id": "b", "name": "Read", "input": {}},
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
            {
                "_type": "UserMessage",
                "content": [
                    {
                        "_type": "ToolResultBlock",
                        "tool_use_id": "a",
                        "is_error": True,
                        "content": "failed",
                    },
                    {
                        "_type": "ToolResultBlock",
                        "tool_use_id": "b",
                        "content": [{"type": "image"}],
                    },
                    {"_type": "TextBlock", "text": "continue"},
                ],
            },
            {
                "_type": "ResultMessage",
                "result": "done",
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "total_cost_usd": 0.1,
            },
        ]
    },
    "killed": {"messages": [{"_type": "AssistantMessage", "content": []}]},
}
agent = object.__new__(ClaudeSDKAgent)
agent.model = "claude-sonnet-5"
agent.version = lambda: "2.1.270"
for name, raw in records.items():
    trajectory = agent.to_trajectory(raw)
    (out / f"{name}.json").write_text(
        json.dumps(
            {
                "record": raw,
                "trajectory": trajectory.to_json_dict(),
                "rendered": render_trajectory(trajectory),
            },
            indent=2,
        )
        + "\n"
    )
