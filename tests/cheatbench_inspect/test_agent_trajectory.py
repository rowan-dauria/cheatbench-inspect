"""Check how bridge messages reach the judge's transcript."""

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall, ToolCallError

from cheatbench_inspect.agent_trajectory import to_trajectory
from cheatbench_inspect.judge import render_trajectory
from cheatbench_inspect.reasoning_replay import pack_reasoning


def test_trajectory_retains_readable_reasoning_and_elides_images() -> None:
    messages = [
        ChatMessageSystem(content="Hidden session setup."),
        ChatMessageUser(content="The task prompt is supplied separately."),
        ChatMessageAssistant(
            content=[
                pack_reasoning(ContentReasoning(reasoning="Check the file.")),
                ContentReasoning(reasoning="encrypted-only", redacted=True),
                ContentText(text="Reading."),
            ],
            tool_calls=[
                ToolCall(id="one", function="Read", arguments={"file_path": "/workspace/file"}),
                ToolCall(id="two", function="Read", arguments={"file_path": "/workspace/image"}),
            ],
        ),
        ChatMessageTool(
            tool_call_id="one",
            content="",
            error=ToolCallError("unknown", "File missing."),
        ),
        ChatMessageTool(
            tool_call_id="two",
            content=[ContentImage(image="data:image/png;base64,c2VjcmV0")],
        ),
        ChatMessageAssistant(content="Finished."),
    ]
    trajectory = to_trajectory(
        messages,
        agent="claude-code",
        version="2.1.270",
        model="claude-sonnet-5",
        details={"tools": ["Read"]},
        final_text="Finished.",
    )
    assert trajectory is not None
    assert (
        render_trajectory(trajectory)
        == """[init] agent=claude-code version=2.1.270 model=claude-sonnet-5 tools=['Read']
THINKING: Check the file.
ASSISTANT: Reading.
[0] TOOL_CALL Read: file_path=/workspace/file
[1] TOOL_CALL Read: file_path=/workspace/image
    -> RESULT[Read] (ERROR): File missing.
    -> RESULT[Read]: <image returned to the model, elided>
ASSISTANT: Finished.
FINAL MESSAGE: Finished."""
    )
