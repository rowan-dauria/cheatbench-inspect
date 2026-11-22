"""Generate Codex runner golden fixtures from original CheatBench 96e05a6.

Runs the original ``core/agents/codex_runner.py`` ``compose_command`` and
``config_toml`` in a read-only subprocess with ``PYTHONDONTWRITEBYTECODE=1``,
``python -B``, and a scratch working directory so the original checkout is
never written to.

Covers Prime default policy (closed web), web on (live), and ``mcp_only``
(chess), plus the bare ``base_url`` cases where the port documents its
``[model_providers.bridge]`` departure.

Usage:

    uv run python scripts/codex_runner_fixtures.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "96e05a6d61faf33be788f75dd01998066045ad02"
DEFAULT_CHEATBENCH = Path("/home/agent/cheatbench")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "cheatbench_inspect"
    / "fixtures"
    / "codex_runner.json"
)

_EVAL_SCRIPT = textwrap.dedent(
    """\
    import json
    import sys
    from unittest.mock import patch

    # Load original codex_runner without mutating real stdout fd 1.
    with open(sys.argv[1], encoding="utf-8") as fh:
        src = fh.read()
    ns = {"__name__": "orig_codex_runner"}
    with patch("os.dup2"):
        exec(compile(src, sys.argv[1], "exec"), ns)

    cases = json.loads(sys.stdin.read())
    out = {
        "compose_cases": [
            {
                **case,
                "expected_command": ns["compose_command"](
                    case["cfg"], case["instruction"], case["logfile"]
                ),
            }
            for case in cases["compose_cases"]
        ],
        "config_cases": [
            {
                **case,
                "expected_original_toml": ns["config_toml"](**case["kwargs"]),
            }
            for case in cases["config_cases"]
        ],
    }
    sys.stdout.write(json.dumps(out))
    """
)


def _build_cases() -> dict[str, list[dict[str, Any]]]:
    compose_cases: list[dict[str, Any]] = [
        {
            "id": "prime_default",
            "cfg": {
                "exec_flags": [
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--skip-git-repo-check",
                    "--strict-config",
                ],
                "model_slug": "gpt-5.6-sol",
                "enable": ["unified_exec"],
                "disable": ["image_generation", "browser_use", "computer_use", "apps"],
                "cli_flags": [
                    "-c",
                    "model_reasoning_effort=high",
                    "-c",
                    "model_reasoning_summary=detailed",
                    "-c",
                    "analytics.enabled=false",
                    "-c",
                    "check_for_update_on_startup=false",
                    "-c",
                    "web_search=disabled",
                ],
            },
            "instruction": "Factor the integer in instance.txt.",
            "logfile": "/tmp/codex-home/codex.txt",
        },
        {
            "id": "web_on",
            "cfg": {
                "exec_flags": [
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--skip-git-repo-check",
                    "--strict-config",
                    "--ignore-user-config",
                ],
                "model_slug": "gpt-5.6-sol",
                "enable": ["unified_exec"],
                "disable": ["image_generation", "browser_use", "computer_use", "apps"],
                "cli_flags": [
                    "-c",
                    "model_reasoning_effort=high",
                    "-c",
                    "analytics.enabled=false",
                    "-c",
                    "check_for_update_on_startup=false",
                    "-c",
                    "web_search=live",
                ],
            },
            "instruction": "Search for the reference paper.",
            "logfile": "/tmp/codex-home/codex.txt",
        },
        {
            "id": "mcp_only_chess",
            "cfg": {
                "exec_flags": [
                    "--skip-git-repo-check",
                    "--strict-config",
                    "--sandbox",
                    "read-only",
                    "-c",
                    'approval_policy="never"',
                ],
                "model_slug": "gpt-5.6-luna",
                "enable": [],
                "disable": [
                    "image_generation",
                    "browser_use",
                    "computer_use",
                    "apps",
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
                ],
                "cli_flags": [
                    "-c",
                    "model_reasoning_effort=high",
                    "-c",
                    "analytics.enabled=false",
                    "-c",
                    "check_for_update_on_startup=false",
                    "-c",
                    "web_search=disabled",
                    "-c",
                    'features.code_mode.direct_only_tool_namespaces=["mcp__chess"]',
                    "-c",
                    "features.code_mode.enabled=false",
                    "-c",
                    "agents.enabled=false",
                ],
            },
            "instruction": "Play chess.",
            "logfile": "/tmp/codex-home/codex.txt",
        },
    ]

    config_cases: list[dict[str, Any]] = [
        {
            "id": "empty_web_on_no_base_url",
            "kwargs": {},
            "provider_block_departure": False,
        },
        {
            "id": "prime_grader_no_base_url",
            "kwargs": {
                "grader_url": "http://127.0.0.1:9000/tok/mcp",
                "grader_name": "verifier",
            },
            "provider_block_departure": False,
        },
        {
            "id": "mcp_only_chess_no_base_url",
            "kwargs": {
                "grader_url": "http://127.0.0.1:9000/tok/mcp",
                "grader_name": "chess",
                "mcp_only": True,
            },
            "provider_block_departure": False,
        },
        {
            "id": "explicit_provider_override",
            "kwargs": {
                "provider": {
                    "id": "override",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "env_key": "OPENAI_API_KEY",
                    "wire_api": "chat",
                },
                "grader_url": "http://127.0.0.1:9000/tok/mcp",
                "grader_name": "verifier",
            },
            "provider_block_departure": False,
        },
        {
            "id": "prime_default_with_base_url",
            "kwargs": {
                "base_url": "http://127.0.0.1:18080/v1",
                "grader_url": "http://127.0.0.1:9000/tok/mcp",
                "grader_name": "verifier",
            },
            "provider_block_departure": True,
        },
        {
            "id": "web_on_with_base_url",
            "kwargs": {
                "base_url": "http://127.0.0.1:18080/v1",
            },
            "provider_block_departure": True,
        },
        {
            "id": "mcp_only_chess_with_base_url",
            "kwargs": {
                "base_url": "http://127.0.0.1:18080/v1",
                "grader_url": "http://127.0.0.1:9000/tok/mcp",
                "grader_name": "chess",
                "mcp_only": True,
            },
            "provider_block_departure": True,
        },
    ]

    return {"compose_cases": compose_cases, "config_cases": config_cases}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Codex runner golden fixtures from original CheatBench."
    )
    parser.add_argument(
        "--cheatbench",
        type=Path,
        default=DEFAULT_CHEATBENCH,
        help=f"Path to a read-only checkout of CheatBench at {UPSTREAM_COMMIT[:7]}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON fixture path.",
    )
    args = parser.parse_args()

    runner_path = args.cheatbench.resolve() / "core" / "agents" / "codex_runner.py"
    cases = _build_cases()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    with tempfile.TemporaryDirectory(prefix="cb-codex-runner-fixtures-") as tmpdir:
        proc = subprocess.run(
            [sys.executable, "-B", "-c", _EVAL_SCRIPT, str(runner_path)],
            input=json.dumps(cases),
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    generated = json.loads(proc.stdout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(generated['compose_cases'])} compose cases and "
        f"{len(generated['config_cases'])} config cases to {args.output}"
    )


if __name__ == "__main__":
    main()
