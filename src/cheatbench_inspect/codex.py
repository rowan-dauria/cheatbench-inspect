# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/core/agents/codex.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.
"""Launch the Codex CLI episode inside the sample sandbox."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from inspect_ai.agent import sandbox_agent_bridge
from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.model._generate_config import (
    active_generate_config,  # inspect-evals-lint: ignore[private_api_imports] -- No public accessor for the active eval config.
)
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from cheatbench_inspect.codex_trajectory import to_trajectory
from cheatbench_inspect.container import AGENT_ENV, AGENT_USER, BOOTSTRAP
from cheatbench_inspect.harness import HarnessPolicy, HarnessResult
from cheatbench_inspect.lost_episodes import (
    MAX_TURNS_EXACT_ERROR,
    REASON_LOST,
    classify,
    is_harness_timeout,
)
from cheatbench_inspect.reasoning_replay import responses_reasoning_filter

_RESOURCES = Path(__file__).parent / "resources"
_DEST_DIR = "/usr/local/lib/codex"
_BIN_LINK = "/usr/local/bin/codex"
_STDERR_SCAN_BYTES = 256 * 1024
# Matches original core/trial.py:1007 (`CLEAN_EXIT_SCAN_LINES`).
_CLEAN_EXIT_SCAN_LINES = 20
_START = """
import io, json, sys
sys.stdin = io.StringIO(json.dumps(PAYLOAD))
exec(compile(PAYLOAD.pop("runner"), "<r>", "exec"), {"__name__": "__main__"})
"""

# Original core/agents/codex.py:176-236, 378-380.
CODEX_HOME = "/tmp/codex-home"
SECRETS_DIR = "/tmp/codex-secrets"
EXEC_FLAGS: tuple[str, ...] = (
    "--dangerously-bypass-approvals-and-sandbox",
    "--skip-git-repo-check",
    "--strict-config",
)
IGNORE_USER_CONFIG = "--ignore-user-config"
ENABLE: tuple[str, ...] = ("unified_exec",)
DISABLE: tuple[str, ...] = ("image_generation", "browser_use", "computer_use", "apps")
MCP_ONLY_DISABLE: tuple[str, ...] = (
    "shell_tool",
    "unified_exec",
    "shell_snapshot",
    "multi_agent",
    "code_mode_host",
    "plugins",
    "remote_plugin",
    "view_image",
    "workspace_dependencies",
    "goals",
    "tool_suggest",
    "enable_request_compression",
)
FIXED_CONFIG: tuple[str, ...] = (
    "analytics.enabled=false",
    "check_for_update_on_startup=false",
)
DEFAULT_REASONING_EFFORT = "high"
REASONING_SUMMARY_CHOICES: tuple[str, ...] = ("auto", "concise", "detailed", "none")
BASE_TOOLS: tuple[str, ...] = ("exec", "apply_patch")
WEB_TOOL = "web__run"

_ASSISTANT_ACTIVITY_TYPES = frozenset(
    {"reasoning", "function_call", "custom_tool_call", "web_search_call"}
)


def _parse_codex_version(stdout: str) -> str:
    """Parse `codex --version` output (`codex-cli X.Y.Z`) as in core/agents/codex.py:160-163."""
    first = stdout.splitlines()[0].strip() if stdout.strip() else ""
    if first.startswith("codex-cli"):
        first = first[len("codex-cli") :].strip()
    return first.split()[0] if first else ""


async def _install_cli(version: str) -> None:
    """Install the pinned Codex package archive into a neutral root-owned location.

    Departure from original ``core/agents/codex.py:257-277`` (npm install): the
    port uses ``inspect_swe``'s checksummed release-archive installer and moves
    the entire extracted package directory (preserving ``bin/codex``,
    ``bin/codex-code-mode-host``, ``codex-path/rg`` and ``codex-resources/bwrap``
    together) to ``/usr/local/lib/codex``, symlinks ``/usr/local/bin/codex``,
    and removes ``inspect_swe``'s ``/var/tmp`` staging directory.
    """
    # inspect_swe has no public binary installer; keep its private API here.
    from inspect_swe._codex_cli.agentbinary import codex_cli_binary_source
    from inspect_swe._util.agentbinary import ensure_agent_binary_installed

    box = sandbox()
    binary = await ensure_agent_binary_installed(codex_cli_binary_source(), version, sandbox=box)
    binary_path = PurePosixPath(binary)
    if binary_path.parent.name != "bin":
        raise RuntimeError(f"Expected packaged Codex binary under bin/, got {binary!r}")
    pkg_dir = str(binary_path.parent.parent)
    staging_dir = str(binary_path.parent.parent.parent)
    cleanup = ' && rm -rf "$3"' if staging_dir not in {"", "/", "/tmp", "/var", "/var/tmp"} else ""
    script = (
        'rm -rf "$2" && mv "$1" "$2" && chown -R 0:0 "$2" && chmod -R a+rX "$2" '
        f'&& ln -sf "$2/bin/codex" {_BIN_LINK}{cleanup} && {_BIN_LINK} --version'
    )
    result = await box.exec(
        ["sh", "-c", script, "sh", pkg_dir, _DEST_DIR, staging_dir],
        user="root",
    )
    if not result.success or _parse_codex_version(result.stdout) != version:
        raise RuntimeError(f"Codex installation failed: {result.stdout} {result.stderr}")


def web_enabled(policy: HarnessPolicy) -> bool:
    """Whether `web__run` is offered. Ported from core/agents/codex.py:280-299."""
    search, fetch = bool(policy.web_search), bool(policy.web_fetch)
    if search != fetch:
        raise ValueError(
            "adapter 'codex' has ONE web knob (`-c web_search=`) and "
            "Codex has ONE web tool (`web__run`, whose operations are "
            "search_query/open/click/find/screenshot), so web_search and "
            "web_fetch cannot differ; this task asks for "
            f"web_search={search}, web_fetch={fetch}"
        )
    return search


def tools_for(policy: HarnessPolicy) -> list[str]:
    """Resolved tool names under `policy`. Ported from core/agents/codex.py:301-307."""
    if policy.mcp_only:
        if web_enabled(policy):
            raise ValueError("mcp_only cannot enable native web tools")
        return []
    return list(BASE_TOOLS) + ([WEB_TOOL] if web_enabled(policy) else [])


def model_slug(model: str, *, custom_base_url: bool = False) -> str:
    """What goes after `--model`. Ported from core/agents/codex.py:384-400."""
    if custom_base_url:
        return model
    return model.split("/")[-1]


def exec_flags(policy: HarnessPolicy, base_url: str = "", *, grader: bool = False) -> list[str]:
    """Non-negotiable `codex exec` flags. Ported from core/agents/codex.py:354-372."""
    tools_for(policy)
    flags = list(EXEC_FLAGS)
    if policy.mcp_only:
        flags.remove("--dangerously-bypass-approvals-and-sandbox")
        flags += ["--sandbox", "read-only", "-c", 'approval_policy="never"']
    if not base_url and not grader:
        flags.append(IGNORE_USER_CONFIG)
    return flags


def enable_features(policy: HarnessPolicy) -> list[str]:
    """`--enable` feature names. Ported from core/agents/codex.py:195,487."""
    tools_for(policy)
    return [] if policy.mcp_only else list(ENABLE)


def disable_features(policy: HarnessPolicy) -> list[str]:
    """`--disable` feature names. Ported from core/agents/codex.py:209,374-381."""
    tools_for(policy)
    disabled = list(DISABLE)
    if policy.mcp_only:
        disabled += list(MCP_ONLY_DISABLE)
    return disabled


def cli_flags(
    policy: HarnessPolicy,
    *,
    generate_config: GenerateConfig | None = None,
    mcp_server: str = "chess",
) -> list[str]:
    """`-c key=value` flags. Ported from core/agents/codex.py:217-226,309-342,412-415.

    Keeps `-c model_reasoning_effort` (defaulting to ``high``) and emits
    `-c model_reasoning_summary` only when ``generate_config.reasoning_summary``
    is set to a value other than ``none``, mirroring the original's model-entry behavior while upstream
    generation parameters remain governed by Inspect's ``GenerateConfig``
    (``forward_generation_config=False``).
    """
    tools_for(policy)
    flags: list[str] = []
    effort = (
        generate_config.reasoning_effort
        if generate_config is not None and generate_config.reasoning_effort is not None
        else DEFAULT_REASONING_EFFORT
    )
    if effort:
        flags += ["-c", f"model_reasoning_effort={effort}"]
    summary = generate_config.reasoning_summary if generate_config is not None else None
    if summary and summary != "none":
        if summary not in REASONING_SUMMARY_CHOICES:
            raise ValueError(
                f"reasoning_summary={summary!r}; codex takes {list(REASONING_SUMMARY_CHOICES)}"
            )
        flags += ["-c", f"model_reasoning_summary={summary}"]
    for setting in FIXED_CONFIG:
        flags += ["-c", setting]
    flags += ["-c", "web_search=" + ("live" if web_enabled(policy) else "disabled")]
    if policy.mcp_only:
        flags += [
            "-c",
            "features.code_mode.direct_only_tool_namespaces=" + json.dumps([f"mcp__{mcp_server}"]),
            "-c",
            "features.code_mode.enabled=false",
            "-c",
            "agents.enabled=false",
        ]
    return flags


def validate_responses_route(model: Model) -> None:
    """Fail loudly if the active Inspect model is not routed over the Responses API."""
    api = model.api
    uses_responses = getattr(api, "uses_responses_api", None)
    if callable(uses_responses):
        is_responses = bool(uses_responses([], model.config))
    else:
        is_responses = bool(getattr(api, "responses_api", False))
    if not is_responses:
        raise ValueError(
            "Codex harness requires a model served over the OpenAI Responses API, "
            f"but active model {model!s} ({type(api).__name__}) is not using the "
            "Responses API. For OpenRouter models, pass model arg `responses_api=true`."
        )


def has_assistant_activity(record: dict[str, Any]) -> bool:
    """Whether a Codex rollout contains any model-produced assistant activity."""
    for event in record.get("messages") or []:
        if not isinstance(event, dict) or event.get("type") != "response_item":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        ptype = payload.get("type")
        if ptype in _ASSISTANT_ACTIVITY_TYPES:
            return True
        if ptype == "message" and payload.get("role") == "assistant":
            return True
    return False


def _payload(
    state: TaskState,
    policy: HarnessPolicy,
    cli_model: str,
    port: int,
    *,
    generate_config: GenerateConfig | None = None,
) -> dict[str, Any]:
    """Build the stdin payload for `resources/codex/episode.py`.

    Departure from original ``core/agents/codex.py:440-555``: staging and
    installers are handled host-side; the bridge URL replaces the upstream
    gateway/forwarder; ``resources/codex/proxy.py`` is shipped as the runner's
    ``proxy`` module for native Responses reasoning replay; and real API keys
    never enter the sandbox (a placeholder key is passed in ``task["env"]``).
    """
    observer = policy.observer
    row: dict[str, Any] = {"id": str(state.sample_id), **state.metadata}
    if observer:
        row.update(observer["row"])
    if row.get("tool_surface") in {"chess", "go"}:
        row = {**row, "episode_timeout_s": policy.timeout_s}

    modules: dict[str, str | None] = {
        "proxy": (_RESOURCES / "codex/proxy.py").read_text(encoding="utf-8"),
        "observer": observer["code"] if observer else None,
    }
    if policy.grader_modules:
        modules.update(policy.grader_modules)

    base_url = f"http://127.0.0.1:{port}/v1"
    has_grader = bool(modules.get("grader"))
    resolved_tools = tools_for(policy)
    mcp_server = str(row.get("tool_surface", "chess"))

    codex_cfg: dict[str, Any] = {
        "model_slug": model_slug(cli_model),
        "mcp_only": policy.mcp_only,
        "exec_flags": exec_flags(policy, base_url, grader=has_grader),
        "enable": enable_features(policy),
        "disable": disable_features(policy),
        "cli_flags": cli_flags(policy, generate_config=generate_config, mcp_server=mcp_server),
        "codex_home": CODEX_HOME,
        "secrets_dir": SECRETS_DIR,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": base_url,
    }

    task: dict[str, Any] = {
        "id": str(state.sample_id),
        "model": cli_model,
        "content": state.input_text,
        "workdir": "/workspace",
        "row": row,
        "staged": observer["staged"] if observer else state.metadata.get("staged_files", []),
        "skip_dirs": list(state.metadata.get("skip_dirs") or ()),
        "deliverable_files": list(row.get("deliverable_files") or ()),
        "tools": resolved_tools,
        "timeout_s": policy.timeout_s,
        "env": {"OPENAI_API_KEY": "placeholder"},
        "codex": codex_cfg,
    }

    return {
        "code": _START,
        "runner": (_RESOURCES / "codex/episode.py").read_text(encoding="utf-8"),
        "task": task,
        "modules": modules,
    }


def _effective_generate_config(model: Model) -> GenerateConfig:
    """Match the bridge merge for the active model."""
    return model.config.merge(active_generate_config())


async def run_codex(
    state: TaskState, policy: HarnessPolicy, *, cli_version: str, cli_model: str
) -> HarnessResult:
    """Install and run the pinned Codex CLI, returning a classified HarnessResult."""
    model = get_model()  # inspect-evals-lint: ignore[IEBP001] -- Binds the active model.
    validate_responses_route(model)
    # The bridge merges the active eval config over the active model config.
    generate_config = _effective_generate_config(model)
    # Validate policy and flags before installing the CLI or starting the bridge.
    cli_flags(policy, generate_config=generate_config)
    await _install_cli(cli_version)
    slug = model_slug(cli_model)
    async with sandbox_agent_bridge(
        model_aliases={slug: model},
        filter=responses_reasoning_filter(),
        forward_generation_config=False,
    ) as bridge:
        result = await sandbox().exec(
            ["python3", "-I", "-c", BOOTSTRAP],
            input=json.dumps(
                _payload(state, policy, cli_model, bridge.port, generate_config=generate_config)
            ),
            timeout=policy.timeout_s + 650,
            user=AGENT_USER,
            env=AGENT_ENV,
        )
    if not result.success:
        raise RuntimeError(result.stderr)
    record = json.loads(result.stdout.splitlines()[-1])
    stderr_tail = result.stderr[-_STDERR_SCAN_BYTES:]
    record["stderr_tail"] = stderr_tail
    error, lost = classify_codex_failure(record, stderr_tail)
    trajectory = (
        to_trajectory(record, cli_version=cli_version, cli_model=slug)
        if has_assistant_activity(record)
        else None
    )
    return HarnessResult(
        trajectory,
        record.get("final_text") or "",
        error,
        record,
        record.get("live_logs"),
        lost,
    )


def classify_codex_failure(record: dict[str, Any], stderr_tail: str) -> tuple[str | None, bool]:
    """Apply the original trial's failure precedence to a native Codex record.

    Mirrors ``core/trial.py:930-999`` (``classify_failure``) and
    ``core/trial.py:1046-1093`` (``lost_on_a_clean_exit``) as well as
    ``cheatbench_inspect.claude_sdk.classify_claude_failure``.
    """
    error = record.get("error")
    if error:
        if (
            isinstance(record.get("install"), dict) and record["install"].get("status") == "failed"
        ) or str(error).startswith("codex install failed:"):
            return error, False
        if str(error).startswith(MAX_TURNS_EXACT_ERROR) or is_harness_timeout(error):
            return error, False
        reason = classify(error, stderr_tail)
        return error, REASON_LOST.get(reason or "", False)
    if not (record.get("final_text") or "").strip():
        lines = [line for line in stderr_tail.splitlines() if line.strip()][
            -_CLEAN_EXIT_SCAN_LINES:
        ]
        if lines:
            reason = classify("\n".join(lines))
            if REASON_LOST.get(reason or "", False):
                return f"codex exited 0 after: {lines[-1].strip()}", True
    return None, False
