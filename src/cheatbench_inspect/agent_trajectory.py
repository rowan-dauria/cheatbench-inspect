"""Render the conversation retained by an Inspect agent for the judge."""

from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentReasoning,
    ContentText,
)

from cheatbench_inspect.reasoning_replay import unpack_reasoning

_IMAGE = "<image returned to the model, elided>"


def message_text(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    return "\n\n".join(
        part.text.strip() if isinstance(part, ContentText) else _IMAGE
        for part in message.content
        if isinstance(part, (ContentText, ContentImage))
    ).strip()


def readable_reasoning(message: ChatMessageAssistant) -> str:
    if isinstance(message.content, str):
        return ""
    blocks = [
        unpack_reasoning(part) for part in message.content if isinstance(part, ContentReasoning)
    ]
    return "\n\n".join(
        text
        for part in blocks
        if (text := (part.summary or ("" if part.redacted else part.reasoning)).strip())
    )


def to_trajectory(
    messages: list[ChatMessage],
    *,
    agent: str,
    version: str,
    model: str,
    details: dict[str, Any],
    final_text: str,
) -> dict[str, Any] | None:
    """Convert assistant turns and associated tool results to ATIF steps."""
    steps: list[dict[str, Any]] = []
    issued: dict[str, dict[str, Any]] = {}
    seen_assistant = False
    for message in messages:
        if isinstance(message, ChatMessageAssistant):
            seen_assistant = True
            step: dict[str, Any] = {"source": "agent", "message": message_text(message)}
            if thinking := readable_reasoning(message):
                step["reasoning_content"] = thinking
            if message.tool_calls:
                step["tool_calls"] = [
                    {
                        "tool_call_id": call.id,
                        "function_name": call.function,
                        "arguments": call.arguments,
                    }
                    for call in message.tool_calls
                ]
                for call in message.tool_calls:
                    issued[call.id] = step
            steps.append(step)
        elif isinstance(message, ChatMessageTool):
            call_id = message.tool_call_id
            result: dict[str, Any] = {"content": message_text(message)}
            if message.error:
                result["extra"] = {"is_error": True}
                if not result["content"]:
                    result["content"] = message.error.message
            if call_id is not None and call_id in issued:
                result["source_call_id"] = call_id
                step = issued[call_id]
            else:
                step = {"source": "user", "message": ""}
                steps.append(step)
            step.setdefault("observation", {"results": []})["results"].append(result)
        elif (
            isinstance(message, ChatMessageUser)
            and seen_assistant
            and (text := message_text(message))
        ):
            steps.append({"source": "user", "message": text})

    if not seen_assistant:
        return None
    for index, step in enumerate(steps, start=1):
        step["step_id"] = index
    return {
        "schema_version": "ATIF-v1.8",
        "agent": {"name": agent, "version": version, "model_name": model, "extra": details},
        "steps": steps,
        "final_metrics": {"total_steps": len(steps)},
        "extra": {"final_text": final_text} if final_text else {},
    }
