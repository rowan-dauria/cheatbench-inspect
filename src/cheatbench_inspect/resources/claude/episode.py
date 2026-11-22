# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/claude_sdk_runner.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.
"""Run the Claude Agent SDK inside the sandbox."""
# pyright: reportMissingImports=false

import asyncio
import dataclasses
import json
import os
import shutil
import sys
import time
from typing import Any


def _log(msg: str) -> None:
    print(f"[runner] {msg}", file=sys.stderr, flush=True)


# Live turn streaming. The host tees these to messages.jsonl / turns.log as they
# arrive, so an episode that is killed or times out still leaves its turns
# behind.
MSG_PREFIX = "\x1eRHMSG "


def _emit_msg(rec: dict) -> None:
    try:
        sys.stderr.write(MSG_PREFIX + json.dumps(rec) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# transcript handling. NO TRUNCATION anywhere: full tool inputs, tool results,
# thinking and text, however long.
# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """SDK messages are dataclasses; json.dumps cannot serialize them."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "_type": type(obj).__name__,
            **{f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
        }
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _elide_images(content):
    """Replace image payloads in a tool result with a note.

    A `Read` of a staged PNG returns the picture base64'd. That payload is
    worthless in a log and bloats every artefact it touches. The picture itself
    is NOT elided anywhere the agent can see -- only in what we persist. Text is
    never touched: this is payload, not content.
    """
    if isinstance(content, list):
        out = []
        for it in content:
            if isinstance(it, dict) and it.get("type") == "image":
                src = it.get("source") or {}
                n = len(src.get("data") or "")
                out.append(
                    {
                        "type": "text",
                        "text": f"<image returned to the model, {n} b64 chars, elided>",
                    }
                )
            else:
                out.append(it)
        return out
    return content


# The redacted-thinking envelope, by either the SDK's class name or the raw API
# block type. OpenRouter reuses it to pass PLAIN reasoning from the models it
# proxies; see the host-side converter in claude_sdk.py, which decodes it.
_REDACTED_TYPES = {"RedactedThinkingBlock", "redacted_thinking"}
_OPENROUTER_REASONING = "openrouter.reasoning:"


def _is_redacted_thinking(b: dict) -> bool:
    return bool(_REDACTED_TYPES & {b.get("_type"), b.get("type")})


def _slim(rec: dict) -> dict | None:
    """Strip a transcript message to what a reader (or the judge) needs."""
    t = rec.get("_type")
    if t == "StreamEvent":
        return None
    if t == "SystemMessage":
        if rec.get("subtype") in {"api_retry", "api_error"}:
            d = rec.get("data") or {}
            # Keep retry telemetry, not request headers, bodies, raw exceptions
            # or a provider's possibly credential-bearing error object.
            safe = {
                k: d[k]
                for k in (
                    "retryAttempt",
                    "maxRetries",
                    "retryInMs",
                    "attempt",
                    "max_retries",
                    "retry_delay_ms",
                    "api_error_status",
                    "status_code",
                )
                if type(d.get(k)) in (int, float)
            }
            return {"_type": t, "subtype": rec["subtype"], "data": safe}
        if rec.get("subtype") != "init":
            return None
        d = rec.get("data") or {}
        return {
            "_type": t,
            "subtype": "init",
            "data": {"tools": d.get("tools"), "mcp_servers": d.get("mcp_servers")},
        }
    if t == "ResultMessage":
        return {
            "_type": t,
            "subtype": rec.get("subtype"),
            "num_turns": rec.get("num_turns"),
            "total_cost_usd": rec.get("total_cost_usd"),
            "usage": rec.get("usage"),
            "result": rec.get("result"),
        }
    blocks = []
    for b in rec.get("content") or []:
        if not isinstance(b, dict):
            continue
        bt = b.get("_type")
        if bt == "TextBlock":
            if (b.get("text") or "").strip():
                blocks.append({"_type": bt, "text": b["text"]})
        elif bt == "ThinkingBlock":
            # `signature` is a multi-KB base64 blob with nothing to read in it,
            # so it is dropped -- but the BLOCK is kept whatever `thinking`
            # holds. Under `display: "omitted"` (the Opus 5 default) the field
            # arrives empty, and a transcript that dropped those blocks read as
            # if the model had never reasoned. It had; the text was withheld.
            # Those are different facts and the transcript must be able to tell
            # them apart. The renderer shows nothing for an empty one.
            blocks.append({"_type": bt, "thinking": b.get("thinking") or ""})
        elif _is_redacted_thinking(b):
            # Anthropic's encrypted-reasoning envelope. `data` is kept ONLY when
            # it is OpenRouter's reuse of the envelope to pass plain reasoning
            # through (`openrouter.reasoning:<b64>`), which the host-side
            # converter decodes; genuine ciphertext is unreadable and is dropped
            # for the same reason `signature` is, leaving the block itself as
            # the record that reasoning happened here.
            out_block = {"_type": b.get("_type") or "redacted_thinking"}
            data = b.get("data")
            if isinstance(data, str) and data.startswith(_OPENROUTER_REASONING):
                out_block["data"] = data
            blocks.append(out_block)
        elif bt == "ToolUseBlock":
            blocks.append(
                {"_type": bt, "id": b.get("id"), "name": b.get("name"), "input": b.get("input")}
            )
        elif bt == "ToolResultBlock":
            blocks.append(
                {
                    "_type": bt,
                    "tool_use_id": b.get("tool_use_id"),
                    "is_error": b.get("is_error"),
                    "content": _elide_images(b.get("content")),
                }
            )
    if not blocks:
        return None
    out = {"_type": t, "content": blocks}
    if rec.get("error"):
        out["error"] = rec["error"]
    return out


def _count_tool_uses(transcript: list[dict]) -> int:
    n = 0
    for msg in transcript:
        if msg.get("_type") != "AssistantMessage":
            continue
        for block in msg.get("content") or []:
            if isinstance(block, dict) and block.get("_type") == "ToolUseBlock":
                n += 1
    return n


def _final_text(transcript: list[dict], result_msg: dict | None) -> str:
    if result_msg and result_msg.get("result"):
        return str(result_msg["result"])
    for msg in reversed(transcript):
        if msg.get("_type") != "AssistantMessage":
            continue
        texts = [
            b.get("text", "")
            for b in (msg.get("content") or [])
            if isinstance(b, dict) and b.get("_type") == "TextBlock"
        ]
        if any(t.strip() for t in texts):
            return "\n".join(texts)
    return ""


# ---------------------------------------------------------------------------
# the episode
# ---------------------------------------------------------------------------
WORKDIR = "/workspace"

# The formats a deliverable can take. An agent that produced its own work writes
# one of these to its cwd; we ship the bytes back to the host so a later pass can
# diff them against whatever was staged.
DELIVERABLE_EXTS = (
    ".xlsx",
    ".xls",
    ".xlsm",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".pdf",
    ".csv",
    ".svg",
)

# Don't ship an unbounded amount back on stdout.
_DELIVERABLE_MAX_BYTES = 64 * 1024 * 1024


def collect_deliverables(root: str, skip_dirs: tuple, include_files: tuple = ()) -> list:
    """Export office files and exact task-requested outputs for the host.

    `skip_dirs` are the top-level directories OUR staging created, so nothing we
    planted is ever re-exported as if the agent had made it. Best effort: a read
    error on one file is skipped, not fatal, since the episode still owes the
    host its JSON line.
    """
    import base64
    import stat

    # Task rows may request exact additional outputs (e.g. a ZIP and message).
    # These names do not broaden collection to other archives or text files.
    requested = {
        p
        for p in include_files
        if isinstance(p, str)
        and p
        and not os.path.isabs(p)
        and not any(part in ("", ".", "..") for part in p.split("/"))
    }
    out, total = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        top = rel_dir.split(os.sep)[0]
        if top in skip_dirs:
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            relative = os.path.relpath(os.path.join(dirpath, name), root)
            if (
                os.path.splitext(name)[1].lower() not in DELIVERABLE_EXTS
                and relative not in requested
            ):
                continue
            q = os.path.join(dirpath, name)
            try:
                # No symlink files; os.walk already excludes symlink directories.
                info = os.lstat(q)
                if not stat.S_ISREG(info.st_mode):
                    continue
                remaining = _DELIVERABLE_MAX_BYTES - total
                if info.st_size > remaining:
                    continue
                with open(q, "rb") as f:
                    data = f.read(remaining + 1)
            except OSError:
                continue
            if total + len(data) > _DELIVERABLE_MAX_BYTES:
                _log(f"deliverable export cap hit; skipping {q}")
                continue
            total += len(data)
            out.append({"name": os.path.relpath(q, root), "b64": base64.b64encode(data).decode()})
    return out


def build_options_kwargs(task: dict, grader_servers: dict, grader_allowed: list) -> dict:
    """Translate the task tool policy into the original SDK options."""
    policy = task.get("tools") or {}
    mcp_only = policy.get("mcp_only", False)
    if mcp_only and (policy.get("web_search") or policy.get("web_fetch")):
        raise ValueError("mcp_only cannot enable native web tools")
    optional = {"web_search": "WebSearch", "web_fetch": "WebFetch"}
    tools = (
        []
        if mcp_only
        else ["Bash", "Read", "Write"] + [name for key, name in optional.items() if policy.get(key)]
    )
    blocked = ["Task", "TodoWrite", "Edit", "NotebookEdit"]
    denied = [name for key, name in optional.items() if not policy.get(key)]
    if mcp_only:
        denied += [
            "Bash",
            "Read",
            "Write",
            "Edit",
            "NotebookEdit",
            "Task",
            "Agent",
            "Glob",
            "Grep",
            "Skill",
            "ToolSearch",
            "TodoWrite",
        ]
    for name in denied:
        if name not in blocked:
            blocked.append(name)
    kwargs = {
        "model": task["model"],
        "cwd": WORKDIR,
        "tools": tools,
        "allowed_tools": [*tools, *grader_allowed],
        "disallowed_tools": blocked,
        "permission_mode": task.get("permission_mode") or "bypassPermissions",
        "mcp_servers": grader_servers,
        "strict_mcp_config": True,
        "setting_sources": [],
        "max_buffer_size": 256 * 1024 * 1024,
        "env": {str(k): str(v) for k, v in (task.get("env") or {}).items()},
        "include_partial_messages": True,
        "cli_path": shutil.which("claude"),
    }
    if task.get("max_turns"):
        kwargs["max_turns"] = task["max_turns"]
    return kwargs


async def run(
    task: dict,
    grader,
    install: dict | None = None,
    forwarder_mod: dict | None = None,
    observer=None,
) -> dict:
    """Run one episode and return the record the host will finish assembling.

    `install` is what `install_agent` reported, carried through to the record.
    `forwarder_mod` is the exec'd core/agents/forwarder.py namespace, present
    only when the model entry set `extra_body`.
    """
    from claude_agent_sdk import ClaudeAgentOptions, query

    model = task["model"]
    timeout_s = int(task.get("timeout_s") or 3600)

    # The task's grader, if it ships one, lives in an in-process MCP server. Its
    # implementation is never written to the container filesystem and never
    # appears in argv or environ, so no shell command the agent can run reaches
    # it -- only its verdicts cross back.
    #
    # A task that ships none arrives here as `({}, [], None)`: no servers, no
    # extra allowed tools, and no grader state to report.
    mcp, extra_allowed, grader_state = grader

    options = ClaudeAgentOptions(**build_options_kwargs(task, mcp, extra_allowed))

    # The prompt was rendered on the host; the container never sees the task
    # folder, the prompt templates or data.jsonl.
    content = task["prompt"]

    async def _input():
        yield {"type": "user", "message": {"role": "user", "content": content}}

    transcript: list[dict] = []
    stream_usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    result_msg: dict | None = None
    init_msg: dict | None = None
    error: str | None = None
    t0 = time.time()

    try:
        async with asyncio.timeout(timeout_s):
            async for message in query(prompt=_input(), options=options):
                raw = _jsonable(message)
                if raw.get("_type") == "StreamEvent":
                    ev = raw.get("event") or {}
                    # message_delta carries the FINAL usage for that turn;
                    # message_start repeats it and would double the input legs.
                    if ev.get("type") == "message_delta":
                        u = ev.get("usage") or {}
                        for _k in (
                            "input_tokens",
                            "output_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens",
                        ):
                            stream_usage[_k] += u.get(_k) or 0
                rec = _slim(raw)
                if rec is None:
                    continue
                transcript.append(rec)
                _emit_msg(rec)
                if observer is not None:
                    observer.observe(rec)
                if rec.get("_type") == "SystemMessage" and rec.get("subtype") == "init":
                    init_msg = rec
                elif rec.get("_type") == "ResultMessage":
                    result_msg = rec
    except TimeoutError:
        error = f"timeout after {timeout_s}s"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        # When the CLI reports an API failure it does so as a ResultMessage
        # whose subtype is "success" and whose `result` is the actual message
        # ("API Error: Connection lost mid-response ..."). The SDK then raises
        # with the SUBTYPE, so the exception reads "returned an error result:
        # success" and carries nothing to classify. We already hold the
        # ResultMessage -- put its text on the record so the failure can be
        # typed instead of landing as `unclassified`.
        res = str((result_msg or {}).get("result") or "")
        if res.startswith("API Error") and res not in error:
            error = f"{error} -- {res}"

    wall = time.time() - t0
    final_text = _final_text(transcript, result_msg)

    init_data = (init_msg or {}).get("data") or {}

    try:
        deliverables = collect_deliverables(
            WORKDIR, tuple(task.get("skip_dirs") or ()), tuple(task.get("deliverable_files") or ())
        )
    except Exception as e:
        _log(f"deliverable collection failed: {type(e).__name__}: {e}")
        deliverables = []

    forwarder_calls = None

    return {
        "id": task["id"],
        "model": model,
        "messages": transcript,
        # The grader's own dict, exactly as the grader kept it.
        "grader_state": grader_state,
        "deliverables": deliverables,
        "final_text": final_text,
        "n_turns": (result_msg or {}).get("num_turns"),
        "n_tool_calls": _count_tool_uses(transcript),
        "wall_time": round(wall, 1),
        "cost_usd": (result_msg or {}).get("total_cost_usd"),
        "stream_usage": stream_usage,
        "usage": (result_msg or {}).get("usage"),
        "result_subtype": (result_msg or {}).get("subtype"),
        "terminal_reason": (result_msg or {}).get("terminal_reason"),
        "session_id": (result_msg or {}).get("session_id"),
        "init_tools": init_data.get("tools"),
        "init_mcp_servers": init_data.get("mcp_servers"),
        "install": install,
        # One summary per request the forwarder relayed (method, path, status,
        # sizes, seconds -- never a body), or None when no forwarder ran.
        "forwarder_calls": forwarder_calls,
        "error": error,
    }


def main() -> None:
    """Execute trusted payload modules in memory and emit one episode record."""
    import contextlib
    import traceback
    import types

    output = sys.stdout
    task: dict = {}
    grader = ({}, [], None)
    observer = None
    with contextlib.redirect_stdout(sys.stderr):
        try:
            payload = json.load(sys.stdin)
            task = payload["task"]
            if task.get("install") or task.get("task_install"):
                raise ValueError("task installers are not supported")
            modules = payload["modules"]
            package = types.ModuleType("_payload")
            package.__path__ = []
            sys.modules[package.__name__] = package
            loaded = {}
            for name in ("verifier", "grader", "observer"):
                if modules.get(name):
                    module = types.ModuleType(f"{package.__name__}.{name}")
                    sys.modules[module.__name__] = module
                    exec(modules[name], module.__dict__)
                    loaded[name] = module
            staged = task.get("staged") or {}
            if "grader" in loaded:
                grader = loaded["grader"].make_grader(task.get("row") or {}, staged, WORKDIR)
            if "observer" in loaded:
                observer = loaded["observer"].MessageObserver(task.get("row") or {}, staged)
            record = asyncio.run(run(task, grader, observer=observer))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            record = {
                "id": task.get("id"),
                "model": task.get("model"),
                "messages": [],
                "grader_state": grader[2],
                "final_text": "",
                "install": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if observer is not None:
            try:
                observer.finish()
            except Exception as exc:
                record["error"] = (
                    f"{record.get('error') or ''} observer finish failed: {exc}".strip()
                )
        record["task_install"] = None
    print(json.dumps(record), file=output, flush=True)


if __name__ == "__main__":
    main()
