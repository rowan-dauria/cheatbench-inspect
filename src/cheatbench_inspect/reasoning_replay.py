"""Replay OpenRouter reasoning through the Anthropic and Responses bridges.

Anthropic bridge: Inspect's anthropic.py:5776-5803 converts unsigned reasoning
to text and loses OpenRouter details. The envelope follows the original
claude_sdk.py:338-386.

Responses bridge: Inspect's responses_impl.py:1494-1513 always serializes
ContentReasoning as a ``<think>`` assistant message rather than a native
``reasoning`` item, and _openai_responses.py:1248 drops ``summary`` whenever
``redacted=True``. The host filter pairs with the in-sandbox Responses proxy
(``resources/codex/proxy.py``) using only public ``GenerateFilter`` hooks.

Remove these once Inspect's bridges replay OpenRouter reasoning natively.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ContentReasoning,
    GenerateConfig,
    GenerateFilter,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from pydantic import ValidationError

_PREFIX = "openrouter.reasoning:"
# Neutral marker keys: both travel inside payloads the sandboxed CLI receives.
_MARKER = "or_reasoning_envelope_v1"
RESPONSES_REASONING_MARKER = "responses_reasoning_v1"
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


def _extract_encrypted(content: ContentReasoning) -> str | None:
    if isinstance(content.internal, dict):
        marker = content.internal.get(RESPONSES_REASONING_MARKER)
        if isinstance(marker, dict):
            enc = marker.get("encrypted_content")
            if isinstance(enc, str) and enc:
                return enc
        stashed = content.internal.get(_REASONING_ENCRYPTED_CONTENT)
        if isinstance(stashed, str) and stashed:
            return stashed
    if content.redacted and content.reasoning:
        return content.reasoning
    return None


def _pack_for_responses_bridge(content: ContentReasoning) -> ContentReasoning:
    enc = _extract_encrypted(content)
    readable_content: str | None = None
    if not content.redacted and content.reasoning:
        readable_content = content.reasoning
    payload: dict[str, Any] = {
        "id": content.signature,
        "summary": content.summary,
        "encrypted_content": enc,
        "content": readable_content,
    }
    base_internal = dict(content.internal) if isinstance(content.internal, dict) else {}
    base_internal[RESPONSES_REASONING_MARKER] = payload
    return content.model_copy(update={"internal": base_internal})


def _prepare_for_responses_upstream(
    content: ContentReasoning,
    seen_by_encrypted: dict[str, dict[str, Any]],
) -> ContentReasoning:
    # Unpack if a packed item ever came back without proxy translation.
    if isinstance(content.internal, dict) and RESPONSES_REASONING_MARKER in content.internal:
        marker = content.internal[RESPONSES_REASONING_MARKER]
        if isinstance(marker, dict):
            enc = marker.get("encrypted_content")
            raw_sig = marker.get("id")
            sig = raw_sig if isinstance(raw_sig, str) else None
            raw_summary = marker.get("summary")
            summary = raw_summary if isinstance(raw_summary, str) else None
            raw_readable = marker.get("content")
            readable = raw_readable if isinstance(raw_readable, str) else ""
            if isinstance(enc, str) and enc:
                if summary:
                    return ContentReasoning(
                        reasoning=readable,
                        summary=summary,
                        signature=sig,
                        redacted=False,
                        internal={_REASONING_ENCRYPTED_CONTENT: enc},
                    )
                return ContentReasoning(
                    reasoning=enc,
                    summary=None,
                    signature=sig,
                    redacted=True,
                    internal=None,
                )

    enc = _extract_encrypted(content)
    cached = seen_by_encrypted.get(enc) if enc else None
    sig = content.signature or (cached["id"] if cached else None)
    summary = content.summary or (cached["summary"] if cached else None)

    if enc:
        if summary:
            # Avoid _openai_responses.py:1248 dropping summary when redacted=True.
            # With redacted=False + internal["reasoning_encrypted_content"], Inspect
            # emits summary=[{"type":"summary_text","text":summary}],
            # encrypted_content=enc, content=[], and id=sig.
            return ContentReasoning(
                reasoning="",
                summary=summary,
                signature=sig,
                redacted=False,
                internal={_REASONING_ENCRYPTED_CONTENT: enc},
            )
        return ContentReasoning(
            reasoning=enc,
            summary=None,
            signature=sig,
            redacted=True,
            internal=None,
        )

    # Plaintext-only reasoning (no encrypted_content): strip marker if any.
    if isinstance(content.internal, dict) and RESPONSES_REASONING_MARKER in content.internal:
        cleaned = {k: v for k, v in content.internal.items() if k != RESPONSES_REASONING_MARKER}
        return content.model_copy(update={"signature": sig, "internal": cleaned or None})
    if sig != content.signature:
        return content.model_copy(update={"signature": sig})
    return content


def responses_reasoning_filter() -> GenerateFilter:
    """Create a fresh per-bridge GenerateFilter for native Responses reasoning replay."""
    seen_by_encrypted: dict[str, dict[str, Any]] = {}

    async def _filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput:
        restored = [
            m.model_copy(
                update={
                    "content": [
                        _prepare_for_responses_upstream(c, seen_by_encrypted)
                        if isinstance(c, ContentReasoning)
                        else c
                        for c in m.content
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
                new_content = []
                for c in message.content:
                    if isinstance(c, ContentReasoning):
                        enc = _extract_encrypted(c)
                        if enc:
                            seen_by_encrypted[enc] = {
                                "id": c.signature,
                                "summary": c.summary,
                            }
                        new_content.append(_pack_for_responses_bridge(c))
                    else:
                        new_content.append(c)
                message = message.model_copy(update={"content": new_content})
            choices.append(choice.model_copy(update={"message": message}))
        return output.model_copy(update={"choices": choices})

    return _filter
