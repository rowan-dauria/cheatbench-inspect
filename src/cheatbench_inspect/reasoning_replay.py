"""Replay OpenRouter reasoning through the Anthropic bridge.

Inspect's anthropic.py:5776-5803 converts unsigned reasoning to text and
loses OpenRouter details. The envelope follows the original claude_sdk.py:
338-386. Remove this once Inspect's bridge replays OpenRouter reasoning natively.
"""

import base64
import binascii
import json

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ContentReasoning,
    GenerateConfig,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from pydantic import ValidationError

_PREFIX = "openrouter.reasoning:"
_MARKER = "cheatbench_inspect_content_reasoning"


def pack_reasoning(content: ContentReasoning) -> ContentReasoning:
    """Wrap reasoning in a thinking signature for lossless bridge replay."""
    summary = content.summary or ("" if content.redacted else content.reasoning)
    envelope = {"type": "reasoning.text", "text": summary, _MARKER: content.model_dump(mode="json")}
    encoded = base64.b64encode(json.dumps(envelope).encode()).decode()
    return ContentReasoning(reasoning=_PREFIX + encoded, summary=summary, redacted=True)


def unpack_reasoning(content: ContentReasoning) -> ContentReasoning:
    """Restore only valid envelopes created by this port."""
    if not content.reasoning.startswith(_PREFIX):
        return content
    try:
        data = json.loads(base64.b64decode(content.reasoning[len(_PREFIX) :], validate=True))
        if not isinstance(data, dict) or _MARKER not in data:
            return content
        return ContentReasoning.model_validate(data[_MARKER])
    except (ValueError, TypeError, binascii.Error, ValidationError):
        return content


async def anthropic_reasoning_filter(
    model: Model,
    messages: list[ChatMessage],
    tools: list[ToolInfo],
    tool_choice: ToolChoice | None,
    config: GenerateConfig,
) -> ModelOutput:
    """Replay wrapped reasoning before generation, and wrap the response."""
    # _model.py:2440-2452 and _bridge/util.py:1338-1345 pass a resolved
    # Model and accept a replacement output; copy inputs and outputs for callers.
    restored = [
        m.model_copy(
            update={
                "content": [
                    unpack_reasoning(c) if isinstance(c, ContentReasoning) else c for c in m.content
                ]
            }
        )
        if isinstance(m, ChatMessageAssistant) and isinstance(m.content, list)
        else m
        for m in messages
    ]
    output = await model.generate(restored, tools=tools, tool_choice=tool_choice, config=config)
    choices = []
    for choice in output.choices:
        message = choice.message
        if isinstance(message.content, list):
            message = message.model_copy(
                update={
                    "content": [
                        pack_reasoning(c) if isinstance(c, ContentReasoning) else c
                        for c in message.content
                    ]
                }
            )
        choices.append(choice.model_copy(update={"message": message}))
    return output.model_copy(update={"choices": choices})
