"""Match CheatBench's judge-facing rendering of a native ATIF trajectory."""

import pytest

from cheatbench_inspect import judge


@pytest.mark.xfail(
    not hasattr(judge, "render_trajectory"),
    raises=AttributeError,
    strict=True,
    reason="Shared trajectory renderer is pending.",
)
def test_render_trajectory_preserves_reasoning_and_paired_tool_results() -> None:
    trajectory = {
        "schema_version": "ATIF-v1.8",
        "session_id": "prime-test",
        "agent": {
            "name": "terminus-2",
            "version": "test",
            "model_name": "openai/gpt-4o",
            "extra": {"parser": "json"},
        },
        "steps": [
            {"step_id": 1, "source": "user", "message": "Factor 15."},
            {
                "step_id": 2,
                "source": "agent",
                "reasoning_content": "  I should read the instance.  ",
                "message": "  Read the instance.  ",
                "tool_calls": [
                    {
                        "tool_call_id": "check",
                        "function_name": "bash_command",
                        "arguments": {"keystrokes": "cat instance.txt", "duration": 0.5},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "check",
                            "content": "NUMBER: 15\n\n\nReady.",
                        }
                    ]
                },
            },
            {"step_id": 3, "source": "agent", "message": "3 times 5."},
        ],
    }

    expected = (
        "[init] agent=terminus-2 version=test model=openai/gpt-4o parser=json\n"
        "    -> USER/SYSTEM[user]: Factor 15.\n"
        "THINKING: I should read the instance.\n"
        "ASSISTANT: Read the instance.\n"
        "[0] TOOL_CALL bash_command: keystrokes=cat instance.txt  duration=0.5\n"
        "    -> RESULT[bash_command]: NUMBER: 15\n\n"
        "Ready.\n"
        "ASSISTANT: 3 times 5."
    )

    assert judge.render_trajectory(trajectory) == expected
