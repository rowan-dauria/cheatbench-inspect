"""Convert Claude SDK records to the ATIF representation used by the judge."""

import base64
import json
from typing import Any


def _metrics(usage):
    if not isinstance(usage, dict):
        return None
    cached = usage.get("cache_read_input_tokens") or 0
    extra = {k: v for k, v in usage.items() if k not in {"input_tokens", "output_tokens"}}
    result = {
        "prompt_tokens": (usage.get("input_tokens") or 0)
        + cached
        + (usage.get("cache_creation_input_tokens") or 0),
        "completion_tokens": usage.get("output_tokens") or 0,
        "cached_tokens": cached,
    }
    if extra:
        result["extra"] = extra
    return result


def _content(raw):
    if raw is None or isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return json.dumps(raw, default=str)
    parts = []
    for item in raw:
        if isinstance(item, dict):
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                value = item["text"]
            elif item.get("type") == "image":
                value = "<image returned to the model, elided>"
            else:
                value = json.dumps(item, default=str)
        else:
            value = str(item)
        parts.append({"type": "text", "text": value})
    return parts or None


def _redacted(block):
    data = block.get("data")
    prefix = "openrouter.reasoning:"
    if not isinstance(data, str) or not data.startswith(prefix):
        return ""
    try:
        inner = json.loads(base64.b64decode(data[len(prefix) :] + "==").decode("utf-8", "replace"))
    except (ValueError, json.JSONDecodeError):
        return ""
    text = inner.get("text") if isinstance(inner, dict) else None
    return text.strip() if isinstance(text, str) else ""


def to_trajectory(record: dict, *, cli_version: str, cli_model: str) -> dict:
    """Return the original converter's serialized ATIF trajectory."""
    steps: list[dict[str, Any]] = []
    issued: dict[str, dict[str, Any]] = {}
    agent_extra = {}
    result_msg = {}
    model = cli_model

    def push(step):
        step["step_id"] = len(steps) + 1
        steps.append(step)
        return step

    for msg in record.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        kind = msg.get("_type")
        if kind == "SystemMessage":
            if msg.get("subtype") == "init" and not agent_extra:
                data = msg.get("data") or {}
                agent_extra = {"tools": data.get("tools"), "mcp_servers": data.get("mcp_servers")}
            continue
        if kind == "ResultMessage":
            result_msg = msg
            continue
        blocks = [b for b in (msg.get("content") or []) if isinstance(b, dict)]
        if kind == "AssistantMessage":
            texts, thinking, calls = [], [], []
            nt = nr = 0
            for block in blocks:
                typ = block.get("_type")
                if typ == "TextBlock" and (block.get("text") or "").strip():
                    texts.append(block["text"].strip())
                elif typ == "ThinkingBlock":
                    nt += 1
                    if (block.get("thinking") or "").strip():
                        thinking.append(block["thinking"].strip())
                elif typ in {"RedactedThinkingBlock", "redacted_thinking"} or block.get("type") in {
                    "RedactedThinkingBlock",
                    "redacted_thinking",
                }:
                    nr += 1
                    plain = _redacted(block)
                    if plain:
                        thinking.append(plain)
                elif typ == "ToolUseBlock":
                    args = block.get("input")
                    calls.append(
                        {
                            "tool_call_id": str(block.get("id") or f"call_{len(issued)}"),
                            "function_name": block.get("name") or "",
                            "arguments": args if isinstance(args, dict) else {"value": args},
                        }
                    )
            step: dict[str, Any] = {
                "source": "agent",
                "message": "\n\n".join(texts),
                "model_name": msg.get("model") or model,
                "llm_call_count": 1,
            }
            if thinking:
                step["reasoning_content"] = "\n\n".join(thinking)
            if calls:
                step["tool_calls"] = calls
            metrics = _metrics(msg.get("usage"))
            if metrics is not None:
                step["metrics"] = metrics
            extra = {}
            if nt:
                extra["thinking_blocks"] = nt
            if nr:
                extra["redacted_thinking_blocks"] = nr
            if extra:
                step["extra"] = extra
            push(step)
            for call in calls:
                issued[call["tool_call_id"]] = step
        elif kind == "UserMessage":
            for block in blocks:
                if block.get("_type") == "ToolResultBlock":
                    call_id = str(block.get("tool_use_id") or "")
                    extra = (
                        dict[str, Any](is_error=True)
                        if block.get("is_error") in (True, "True")
                        else {}
                    )
                    observation = {}
                    if call_id in issued:
                        observation["source_call_id"] = call_id
                        step = issued[call_id]
                    else:
                        extra["tool_use_id"] = call_id
                        step = push({"source": "user", "message": ""})
                    content = _content(block.get("content"))
                    if content is not None:
                        observation["content"] = content
                    if extra:
                        observation["extra"] = extra
                    step.setdefault("observation", {"results": []})["results"].append(observation)
                elif block.get("_type") == "TextBlock" and (block.get("text") or "").strip():
                    push({"source": "user", "message": block["text"].strip()})
    if not steps:
        push(
            {
                "source": "system",
                "message": "(no messages were captured for this episode)",
                "extra": {"placeholder": True},
            }
        )
    usage = result_msg.get("usage") or record.get("usage")
    totals = _metrics(usage)
    final: dict[str, Any] = {"total_steps": len(steps)}
    if totals:
        final.update(
            total_prompt_tokens=totals["prompt_tokens"],
            total_completion_tokens=totals["completion_tokens"],
            total_cached_tokens=totals["cached_tokens"],
        )
    cost = result_msg.get("total_cost_usd") or record.get("cost_usd")
    if cost is not None:
        final["total_cost_usd"] = cost
    if usage:
        final["extra"] = {"usage": usage}
    agent: dict[str, Any] = {
        "name": "claude-sdk",
        "version": cli_version or "unknown",
        "model_name": model,
    }
    if agent_extra:
        agent["extra"] = agent_extra
    trajectory = {
        "schema_version": "ATIF-v1.8",
        "agent": agent,
        "steps": steps,
        "final_metrics": final,
    }
    if record.get("session_id") is not None:
        trajectory["session_id"] = record["session_id"]
    final_text = (
        str(result_msg["result"]) if result_msg.get("result") else record.get("final_text") or ""
    )
    if final_text:
        trajectory["extra"] = {"final_text": final_text}
    return trajectory
