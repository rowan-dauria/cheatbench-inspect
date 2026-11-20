"""Match CheatBench's judge-facing rendering of a native ATIF trajectory."""

import pytest

from cheatbench_inspect import judge


@pytest.mark.parametrize(
    ("optional_fields", "suffix"),
    [
        ({}, ""),
        (
            {"model_name": "openai/gpt-4o", "extra": {"parser": "json"}},
            " model=openai/gpt-4o parser=json",
        ),
    ],
    ids=["required-fields", "with-model-and-extra"],
)
def test_render_trajectory_header(optional_fields: dict, suffix: str) -> None:
    trajectory = {
        "agent": {"name": "terminus-2", "version": "test", **optional_fields},
        "steps": [],
    }

    assert judge.render_trajectory(trajectory) == f"[init] agent=terminus-2 version=test{suffix}"


def test_render_trajectory_messages() -> None:
    trajectory = {
        "agent": {"name": "terminus-2", "version": "test"},
        "steps": [
            {"source": "user", "message": "  Factor 15.  "},
            {"source": "system", "message": "Read the instructions."},
            {
                "source": "agent",
                "reasoning_content": "  Consider 3.\n\n\nCheck 5.  ",
                "message": "  The pair is 3 and 5.  ",
            },
            {"source": "agent", "reasoning_content": "  ", "message": "\n "},
        ],
    }

    assert judge.render_trajectory(trajectory) == (
        "[init] agent=terminus-2 version=test\n"
        "    -> USER/SYSTEM[user]: Factor 15.\n"
        "    -> USER/SYSTEM[system]: Read the instructions.\n"
        "THINKING: Consider 3.\n\n"
        "Check 5.\n"
        "ASSISTANT: The pair is 3 and 5."
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


def test_render_trajectory_preserves_context_and_elides_observation_media() -> None:
    image = {"type": "image", "source": {"media_type": "image/png", "path": "figure.png"}}
    trajectory = {
        "schema_version": "ATIF-v1.8",
        "session_id": "media-test",
        "agent": {"name": "terminus-2", "version": "test"},
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "model_name": "first",
                "message": "",
            },
            {
                "step_id": 2,
                "source": "agent",
                "model_name": "second",
                "reasoning_effort": "high",
                "message": [{"type": "text", "text": "Inspect the figure."}, image],
                "tool_calls": [
                    {
                        "tool_call_id": "open",
                        "function_name": "view_images",
                        "arguments": {"paths": ["figure.png"]},
                    },
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "open",
                            "content": [{"type": "text", "text": "Preview"}, image],
                            "extra": {"is_error": True},
                        },
                        {"content": "A" * 200},
                    ],
                },
            },
        ],
        "extra": {"final_text": "Done."},
    }

    assert judge.render_trajectory(trajectory) == (
        "[init] agent=terminus-2 version=test\n"
        "[context] model=second reasoning_effort=high\n"
        "ASSISTANT: Inspect the figure.\n"
        "<image: figure.png>\n"
        '[0] TOOL_CALL view_images: paths=["figure.png"]\n'
        "    -> RESULT[view_images] (ERROR): Preview\n"
        "<image returned to the model, elided>\n"
        "    -> RESULT[tool]: <base64 blob, 200 chars, elided>\n"
        "FINAL MESSAGE: Done."
    )
