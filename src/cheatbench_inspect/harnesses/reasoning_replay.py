"""Preserve readable OpenRouter reasoning across the agent bridges.

Anthropic transports the reasoning in a thinking signature. The Responses
bridge transports it in a think tag and restores the reasoning object on
replay. The Responses provider drops summaries from redacted reasoning, so
prepare a copy with the encrypted content and summary in separate fields.
"""

from __future__ import annotations

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
# Neutral marker keys: both travel inside payloads the sandboxed CLI receives.
_MARKER = "or_reasoning_envelope_v1"
_REASONING_ENCRYPTED_CONTENT = "reasoning_encrypted_content"


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


def _with_responses_summary(content: ContentReasoning) -> ContentReasoning:
    if content.redacted and content.summary:
        return content.model_copy(
            update={
                "reasoning": "",
                "redacted": False,
                "internal": {
                    **(content.internal if isinstance(content.internal, dict) else {}),
                    _REASONING_ENCRYPTED_CONTENT: content.reasoning,
                },
            }
        )
    return content


async def responses_reasoning_filter(
    model: Model,
    messages: list[ChatMessage],
    tools: list[ToolInfo],
    tool_choice: ToolChoice | None,
    config: GenerateConfig,
) -> ModelOutput:
    """Preserve the bridge-restored summary and encrypted content upstream."""
    restored = [
        message.model_copy(
            update={
                "content": [
                    _with_responses_summary(part) if isinstance(part, ContentReasoning) else part
                    for part in message.content
                ]
            }
        )
        if isinstance(message, ChatMessageAssistant) and isinstance(message.content, list)
        else message
        for message in messages
    ]
    return await model.generate(restored, tools=tools, tool_choice=tool_choice, config=config)
