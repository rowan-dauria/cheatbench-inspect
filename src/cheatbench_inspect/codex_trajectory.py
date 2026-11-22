# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/codex.py
# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/base.py
# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/trajectory/
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.
"""Convert Codex CLI rollout records to the ATIF representation used by the judge."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION: Literal["ATIF-v1.8"] = "ATIF-v1.8"
_AGENT_NAME = "codex"
_EMPTY_ROLLOUT_NOTE = "(no rollout events were captured for this episode)"

_AudioMediaType = Literal[
    "audio/wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "audio/aiff",
]

_AUDIO_MEDIA_TYPE_ALIASES = {
    "audio/mp3": "audio/mpeg",
    "audio/mpga": "audio/mpeg",
    "audio/x-mpeg": "audio/mpeg",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/vnd.wave": "audio/wav",
    "audio/x-m4a": "audio/mp4",
    "audio/m4a": "audio/mp4",
    "audio/x-aac": "audio/aac",
    "audio/x-flac": "audio/flac",
    "audio/x-aiff": "audio/aiff",
}


class ImageSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    path: str


class AudioSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media_type: _AudioMediaType
    path: str
    duration_sec: float | None = Field(default=None, ge=0)

    @field_validator("media_type", mode="before")
    @classmethod
    def normalize_media_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip().lower()
            return _AUDIO_MEDIA_TYPE_ALIASES.get(normalized, normalized)
        return value


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image", "audio"]
    text: str | None = None
    source: ImageSource | AudioSource | None = None

    @model_validator(mode="after")
    def validate_content_type(self) -> ContentPart:
        if self.type == "text":
            if self.text is None:
                raise ValueError("'text' field is required when type='text'")
            if self.source is not None:
                raise ValueError("'source' field is not allowed when type='text'")
            return self
        if self.source is None:
            raise ValueError(f"'source' field is required when type='{self.type}'")
        if self.text is not None:
            raise ValueError(f"'text' field is not allowed when type='{self.type}'")
        expected = ImageSource if self.type == "image" else AudioSource
        if not isinstance(self.source, expected):
            raise ValueError(
                f"type='{self.type}' requires a {expected.__name__}, but the "
                f"source's media_type is {self.source.media_type!r}"
            )
        return self


class Agent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]
    extra: dict[str, Any] | None = None


class ObservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_call_id: str | None = None
    content: str | list[ContentPart] | None = None
    extra: dict[str, Any] | None = None


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[ObservationResult]


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int = Field(ge=1)
    timestamp: str | None = None
    source: Literal["system", "user", "agent"]
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    message: str | list[ContentPart]
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    observation: Observation | None = None
    metrics: Metrics | None = None
    is_copied_context: bool | None = None
    llm_call_count: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                datetime.fromisoformat(v)
            except ValueError as e:
                raise ValueError(f"Invalid ISO 8601 timestamp: {e}") from e
        return v

    @model_validator(mode="after")
    def validate_agent_only_fields(self) -> Step:
        if self.source != "agent":
            for field in (
                "model_name",
                "reasoning_effort",
                "reasoning_content",
                "tool_calls",
                "metrics",
            ):
                if getattr(self, field) is not None:
                    raise ValueError(
                        f"Field '{field}' is only applicable when source is 'agent', "
                        f"but source is '{self.source}'"
                    )
        return self

    @model_validator(mode="after")
    def validate_llm_call_count_zero_fields(self) -> Step:
        if self.llm_call_count == 0 and self.source == "agent":
            for field in ("metrics", "reasoning_content"):
                if getattr(self, field) is not None:
                    raise ValueError(
                        f"Field '{field}' must be absent when llm_call_count is 0 "
                        f"(deterministic dispatch on a 'source: agent' step)"
                    )
        return self


class FinalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] | None = None


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "ATIF-v1.0",
        "ATIF-v1.1",
        "ATIF-v1.2",
        "ATIF-v1.3",
        "ATIF-v1.4",
        "ATIF-v1.5",
        "ATIF-v1.6",
        "ATIF-v1.7",
        "ATIF-v1.8",
    ] = SCHEMA_VERSION
    session_id: str | None = None
    trajectory_id: str | None = None
    agent: Agent
    steps: list[Step] = Field(min_length=1)
    notes: str | None = None
    final_metrics: FinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    extra: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_step_ids(self) -> Trajectory:
        for i, step in enumerate(self.steps):
            expected_step_id = i + 1
            if step.step_id != expected_step_id:
                raise ValueError(
                    f"steps[{i}].step_id: expected {expected_step_id} "
                    f"(sequential from 1), got {step.step_id}"
                )
        return self

    @model_validator(mode="after")
    def validate_tool_call_references(self) -> Trajectory:
        for step in self.steps:
            if step.observation is None:
                continue
            tool_call_ids = (
                {tc.tool_call_id for tc in step.tool_calls} if step.tool_calls else set()
            )
            for result in step.observation.results:
                if result.source_call_id is not None and result.source_call_id not in tool_call_ids:
                    raise ValueError(
                        f"Observation result references source_call_id "
                        f"'{result.source_call_id}' which is not found in "
                        f"step {step.step_id}'s tool_calls"
                    )
        return self

    def to_json_dict(self, exclude_none: bool = True) -> dict[str, Any]:
        return self.model_dump(exclude_none=exclude_none, mode="json")


def _iso(timestamp: Any) -> str | None:
    """Return a rollout timestamp when valid ISO 8601, otherwise None."""
    if not isinstance(timestamp, str):
        return None
    try:
        datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    return timestamp


def _message_text(content: Any) -> str:
    """Extract concatenated text from a Codex message content list."""
    parts = []
    for block in content or []:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def _output_blob(raw: Any) -> tuple[Any, dict[str, Any] | None]:
    """Parse (text, metadata) from a tool output payload."""
    if raw is None:
        return None, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw, None
    else:
        parsed = raw
    if isinstance(parsed, dict):
        out = parsed.get("output")
        if out is None and parsed:
            out = json.dumps(parsed, ensure_ascii=False)
        meta = parsed.get("metadata")
        return out, meta if isinstance(meta, dict) else None
    if isinstance(parsed, list):
        return (
            "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in parsed
            ),
            None,
        )
    return str(parsed), None


def _metrics_from_token_count(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract per-call ATIF metrics from a Codex token_count payload."""
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    last = info.get("last_token_usage")
    if not isinstance(last, dict):
        return None
    extra = {
        "reasoning_output_tokens": last.get("reasoning_output_tokens"),
        "total_tokens": last.get("total_tokens"),
    }
    cache_write = last.get("cache_write_input_tokens")
    if cache_write is not None:
        extra["cache_write_input_tokens"] = cache_write
    prompt = last.get("input_tokens")
    return {
        "prompt_tokens": prompt or None,
        "completion_tokens": last.get("output_tokens") or None,
        "cached_tokens": last.get("cached_input_tokens") or None,
        "extra": extra,
    }


def _group_events_by_api_call_id(normalized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bundle assistant events sharing an api_call_id into single ATIF steps."""
    result: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def add_reasoning(group: dict[str, Any], event: dict[str, Any]) -> None:
        text = event.get("reasoning")
        if not text:
            return
        group["reasoning"] = text if not group.get("reasoning") else f"{group['reasoning']}\n{text}"

    def flush() -> None:
        for gid in order:
            group = groups.pop(gid, None)
            if group is None:
                continue
            group["tool_calls"].sort(key=lambda tc: tc.get("tool_order", 0))
            group["text"] = "\n\n".join(
                p for p in group.pop("message_parts") if isinstance(p, str) and p
            )
            result.append(group)
        order.clear()

    for event in normalized:
        api_call_id = event.get("api_call_id")
        kind = event.get("kind")

        if kind == "message" and event.get("role") != "assistant":
            flush()
            result.append(event)
            continue
        if kind == "observation":
            flush()
            result.append(event)
            continue
        if not isinstance(api_call_id, str):
            flush()
            result.append(event)
            continue

        if api_call_id not in groups:
            groups[api_call_id] = {
                "kind": "bundled",
                "api_call_id": api_call_id,
                "codex_turn_id": event.get("codex_turn_id"),
                "timestamp": event.get("timestamp"),
                "message_parts": [],
                "reasoning": None,
                "tool_calls": [],
                "metrics": event.get("metrics"),
            }
            order.append(api_call_id)

        group = groups[api_call_id]
        if kind == "message":
            text = event.get("text")
            if isinstance(text, str) and text:
                group["message_parts"].append(text)
            add_reasoning(group, event)
            if event.get("timestamp"):
                group["timestamp"] = event["timestamp"]
        elif kind == "tool_call":
            group["tool_calls"].append(event)
            add_reasoning(group, event)
            if not group.get("metrics") and event.get("metrics"):
                group["metrics"] = event["metrics"]

    flush()
    return result


def _event_to_step(event: dict[str, Any], step_id: int, model: str | None) -> Step:
    """Convert one grouped Codex rollout event to an ATIF Step."""
    kind = event.get("kind")
    timestamp = _iso(event.get("timestamp"))

    if kind == "message":
        role = event.get("role", "user")
        source: Literal["system", "user", "agent"] = (
            "agent" if role == "assistant" else "user" if role == "user" else "system"
        )
        return Step(
            step_id=step_id,
            timestamp=timestamp,
            source=source,
            message=event.get("text", ""),
            reasoning_content=event.get("reasoning") if source == "agent" else None,
            model_name=model if source == "agent" else None,
            llm_call_count=1 if source == "agent" else None,
        )

    if kind == "observation":
        return Step(
            step_id=step_id,
            timestamp=timestamp,
            source="agent",
            message="",
            observation=Observation(
                results=[ObservationResult(source_call_id=None, content=event.get("output"))]
            ),
        )

    if kind == "bundled":
        calls: list[ToolCall] = []
        results: list[ObservationResult] = []
        details: dict[str, Any] = {}
        for spec in event.get("tool_calls", []):
            call_id = spec.get("call_id") or ""
            args = spec.get("arguments") or {}
            if not isinstance(args, dict):
                args = {"value": args}
            calls.append(
                ToolCall(
                    tool_call_id=call_id,
                    function_name=spec.get("tool_name") or "",
                    arguments=args,
                )
            )
            if "output" in spec:
                results.append(
                    ObservationResult(
                        source_call_id=call_id or None,
                        content=spec.get("output"),
                    )
                )
            detail = {
                k: spec.get(k)
                for k in ("metadata", "raw_arguments", "item_type", "status")
                if spec.get(k)
            }
            if detail:
                details[call_id] = detail

        extra: dict[str, Any] = {}
        for key in ("api_call_id", "codex_turn_id"):
            if event.get(key):
                extra[key] = event[key]
        if details:
            extra["tool_call_details"] = details

        return Step(
            step_id=step_id,
            timestamp=timestamp,
            source="agent",
            message=event.get("text", ""),
            model_name=model,
            reasoning_content=event.get("reasoning") or None,
            tool_calls=calls or None,
            observation=Observation(results=results) if results else None,
            metrics=Metrics(**event["metrics"]) if event.get("metrics") else None,
            llm_call_count=1,
            extra=extra or None,
        )

    raise ValueError(f"Unsupported event kind '{kind}'")


def _empty_step(note: str) -> Step:
    """Return the ATIF placeholder step for an episode with no captured steps."""
    return Step(step_id=1, source="system", message=note, extra={"placeholder": True})


def to_trajectory(
    record: dict[str, Any] | None,
    *,
    cli_version: str | None,
    cli_model: str | None,
) -> dict[str, Any]:
    """Return the original Codex converter's serialized ATIF trajectory."""
    events = [e for e in ((record or {}).get("messages") or []) if isinstance(e, dict)]

    session_meta = next((e for e in events if e.get("type") == "session_meta"), None)
    meta = (session_meta or {}).get("payload") or {}
    session_id = meta.get("id") or (record or {}).get("session_id")

    agent_version = meta.get("cli_version") or cli_version or "unknown"
    agent_extra = {k: meta[k] for k in ("originator", "cwd") if meta.get(k) is not None} or None

    model: str | None = None
    for event in events:
        if event.get("type") == "turn_context":
            candidate = (event.get("payload") or {}).get("model")
            if isinstance(candidate, str):
                model = candidate
                break
    model = model or cli_model

    # -- normalise every rollout line into one flat event stream -------------
    normalized: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    reasoning: str | None = None
    turn_id: str | None = None
    api_index = 1
    api_call_id = f"api_call_{api_index}"
    api_metrics: dict[str, dict[str, Any]] = {}
    saw_output = False
    tool_order = 0

    def finish_api_call(payload: dict[str, Any]) -> None:
        nonlocal api_index, api_call_id, saw_output, tool_order
        if not saw_output:
            return
        metrics = _metrics_from_token_count(payload)
        if metrics:
            api_metrics[api_call_id] = metrics
        api_index += 1
        api_call_id = f"api_call_{api_index}"
        saw_output = False
        tool_order = 0

    for event in events:
        etype = event.get("type")
        raw_payload = event.get("payload")
        payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
        timestamp = event.get("timestamp")

        if etype == "event_msg":
            kind = payload.get("type")
            if kind in {"task_started", "turn_started"}:
                tid = payload.get("turn_id")
                turn_id = tid if isinstance(tid, str) else None
            elif kind in {"task_complete", "turn_complete", "turn_aborted"}:
                turn_id = None
            elif kind == "token_count":
                finish_api_call(payload)
            continue

        if etype == "turn_context":
            tid = payload.get("turn_id")
            if isinstance(tid, str) and turn_id is None:
                turn_id = tid
            continue

        if etype != "response_item":
            continue

        ptype = payload.get("type")

        if ptype == "reasoning":
            summary = payload.get("summary")
            parts = []
            for item in summary if isinstance(summary, list) else []:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                chunk = "\n".join(parts)
                reasoning = chunk if reasoning is None else f"{reasoning}\n{chunk}"
            continue

        if ptype == "message":
            role = payload.get("role", "user")
            content = payload.get("content")
            normalized.append(
                {
                    "kind": "message",
                    "api_call_id": api_call_id,
                    "codex_turn_id": turn_id,
                    "timestamp": timestamp,
                    "role": role,
                    "text": _message_text(content) if isinstance(content, list) else "",
                    "reasoning": reasoning if role == "assistant" else None,
                }
            )
            if role == "assistant":
                saw_output = True
            reasoning = None
            continue

        if ptype == "web_search_call":
            action = payload.get("action") or {}
            args: Any = {"action_type": action.get("type", "")}
            for key in ("query", "queries", "url"):
                if key in action:
                    args[key] = action[key]
            normalized.append(
                {
                    "kind": "tool_call",
                    "api_call_id": api_call_id,
                    "codex_turn_id": turn_id,
                    "tool_order": tool_order,
                    "timestamp": timestamp,
                    "call_id": "",
                    "tool_name": "web_search_call",
                    "arguments": args,
                    "raw_arguments": None,
                    "reasoning": reasoning,
                    "status": payload.get("status"),
                    "output": (
                        "(results are returned to the model inside the "
                        "provider's response and are not recorded in the "
                        "rollout)"
                    ),
                }
            )
            tool_order += 1
            saw_output = True
            reasoning = None
            continue

        if ptype in {"function_call", "custom_tool_call"}:
            call_id = payload.get("call_id")
            if not call_id:
                continue
            raw_args = payload.get("arguments" if ptype == "function_call" else "input")
            try:
                args = json.loads(raw_args)  # type: ignore[arg-type]
            except (json.JSONDecodeError, TypeError):
                args = None
            if args is None:
                if isinstance(raw_args, str):
                    args = {"input": raw_args}
                elif raw_args is None:
                    args = {}
                else:
                    args = {"value": raw_args}
            pending[call_id] = {
                "kind": "tool_call",
                "api_call_id": api_call_id,
                "codex_turn_id": turn_id,
                "tool_order": tool_order,
                "timestamp": timestamp,
                "call_id": call_id,
                "tool_name": payload.get("name") or "",
                "arguments": args,
                "raw_arguments": raw_args,
                "item_type": ptype,
                "reasoning": reasoning,
                "status": payload.get("status"),
            }
            tool_order += 1
            saw_output = True
            reasoning = None
            continue

        if ptype in {"function_call_output", "custom_tool_call_output"}:
            call_id = payload.get("call_id")
            output, metadata = _output_blob(payload.get("output"))
            info = pending.pop(call_id, None) if call_id else None
            if info is None:
                normalized.append(
                    {
                        "kind": "observation",
                        "api_call_id": api_call_id,
                        "codex_turn_id": turn_id,
                        "tool_order": tool_order,
                        "timestamp": timestamp,
                        "call_id": call_id or "",
                        "output": output,
                        "metadata": metadata,
                    }
                )
                tool_order += 1
                reasoning = None
                continue
            info["output"] = output
            info["metadata"] = metadata
            info["timestamp"] = info.get("timestamp") or timestamp
            normalized.append(info)
            reasoning = None
            continue

    for info in pending.values():
        normalized.append(info)
    pending.clear()

    for event in normalized:
        key = event.get("api_call_id")
        if isinstance(key, str) and key in api_metrics:
            event["metrics"] = api_metrics[key]

    steps: list[Step] = []
    dropped: list[str] = []
    for event in _group_events_by_api_call_id(normalized):
        try:
            steps.append(_event_to_step(event, len(steps) + 1, model))
        except ValueError as e:
            dropped.append(f"{event.get('kind')}: {type(e).__name__}: {e}")

    if not steps:
        steps = [_empty_step(_EMPTY_ROLLOUT_NOTE)]

    # -- totals, from the LAST token_count that carried them -----------------
    final_metrics: FinalMetrics | None = None
    for event in reversed(events):
        if event.get("type") != "event_msg":
            continue
        raw_payload = event.get("payload")
        if not isinstance(raw_payload, dict) or raw_payload.get("type") != "token_count":
            continue
        info = raw_payload.get("info")
        if not isinstance(info, dict):
            continue
        total = info.get("total_token_usage")
        if not isinstance(total, dict):
            continue
        extra: dict[str, Any] = {
            "reasoning_output_tokens": total.get("reasoning_output_tokens"),
            "total_tokens": total.get("total_tokens"),
            "last_token_usage": info.get("last_token_usage"),
        }
        cache_write = total.get("cache_write_input_tokens")
        if cache_write is not None:
            extra["total_cache_write_input_tokens"] = cache_write
        prompt = total.get("input_tokens")
        final_metrics = FinalMetrics(
            total_prompt_tokens=prompt or None,
            total_completion_tokens=total.get("output_tokens") or None,
            total_cached_tokens=total.get("cached_input_tokens") or None,
            total_cost_usd=(
                info.get("total_cost")
                if info.get("total_cost") is not None
                else (record or {}).get("cost_usd")
            ),
            total_steps=len(steps),
            extra=extra,
        )
        break

    final_text = (record or {}).get("final_text") or ""
    if not final_text:
        for step in reversed(steps):
            text = step.message if isinstance(step.message, str) else ""
            if step.source == "agent" and text.strip():
                final_text = text
                break

    trajectory = Trajectory(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        agent=Agent(
            name=_AGENT_NAME,
            version=agent_version,
            model_name=model,
            extra=agent_extra,
        ),
        steps=steps,
        notes=(
            f"{len(dropped)} rollout event(s) could not be converted to "
            f"ATIF steps and are missing from this trajectory: " + "; ".join(dropped)
        )
        if dropped
        else None,
        final_metrics=final_metrics,
        extra={"final_text": final_text} if final_text else None,
    )
    return trajectory.to_json_dict()
