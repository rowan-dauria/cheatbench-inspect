"""Generate Codex launcher golden fixtures from original CheatBench 96e05a6.

Instantiates the original ``core.agents.codex.CodexAgent`` across versions
``0.154.0``, ``0.156.1``, and ``0.159.1`` crossed with all supported policies,
generation settings, and invalid policy combinations in a read-only subprocess
with ``PYTHONDONTWRITEBYTECODE=1``, ``python -B``, and a scratch working
directory so the original checkout is never written to.

Usage:

    uv run python scripts/codex_launcher_fixtures.py
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
    / "codex_launcher.json"
)
VERSIONS = ("0.154.0", "0.156.1", "0.159.1")

_EVAL_SCRIPT = textwrap.dedent(
    """\
    import json
    import sys

    sys.path.insert(0, sys.argv[1])
    from core.agents.codex import CodexAgent

    spec = json.loads(sys.stdin.read())
    valid_results = []
    for case in spec["valid_cases"]:
        agent = CodexAgent(
            model=case["model"],
            generation_config=case["generation_config"],
            version=case["version"],
            base_url=case.get("base_url"),
            api_base_url=case.get("api_base_url"),
        )
        agent.apply_tool_policy(case["policy"])
        blob = json.loads(
            agent.blob(
                row=case["row"],
                prompt="Test prompt.",
                files=[],
                skip_dirs=(),
                grader_src="def make_tool(*a): pass" if case["grader"] else "",
                stage_src="",
                timeout_s=3600,
                env={},
            ).decode("utf-8")
        )
        codex_cfg = blob["task"]["codex"]
        valid_results.append(
            {
                **case,
                "expected": {
                    "model_slug": codex_cfg["model_slug"],
                    "tools": blob["task"]["tools"],
                    "exec_flags": codex_cfg["exec_flags"],
                    "enable": codex_cfg["enable"],
                    "disable": codex_cfg["disable"],
                    "cli_flags": codex_cfg["cli_flags"],
                },
            }
        )

    invalid_results = []
    for case in spec["invalid_cases"]:
        agent = CodexAgent(
            model=case["model"],
            generation_config=case["generation_config"],
            version=case["version"],
        )
        try:
            agent.apply_tool_policy(case["policy"])
        except Exception as exc:
            invalid_results.append(
                {
                    **case,
                    "expected_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
        else:
            raise AssertionError(f"Expected error for invalid case {case['id']}")

    sys.stdout.write(
        json.dumps(
            {
                "upstream_commit": spec["upstream_commit"],
                "valid_cases": valid_results,
                "invalid_cases": invalid_results,
            }
        )
    )
    """
)


def _build_spec() -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = [
        {
            "scenario": "prime_default_effort_high",
            "model": "openai/gpt-5.6-luna",
            "generation_config": {"reasoning_effort": "high"},
            "policy": {"web_search": False, "web_fetch": False, "mcp_only": False},
            "base_url": None,
            "api_base_url": "http://127.0.0.1:13131/v1",
            "grader": True,
            "row": {"id": "pf-1"},
        },
        {
            "scenario": "prime_default_effort_high_summary_detailed",
            "model": "openai/gpt-5.6-sol",
            "generation_config": {"reasoning_effort": "high", "reasoning_summary": "detailed"},
            "policy": {"web_search": False, "web_fetch": False, "mcp_only": False},
            "base_url": None,
            "api_base_url": "http://127.0.0.1:13131/v1",
            "grader": True,
            "row": {"id": "pf-2"},
        },
        {
            "scenario": "prime_default_empty_gen_config",
            "model": "openai/gpt-5",
            "generation_config": {},
            "policy": {"web_search": False, "web_fetch": False, "mcp_only": False},
            "base_url": None,
            "api_base_url": "http://127.0.0.1:13131/v1",
            "grader": True,
            "row": {"id": "pf-3"},
        },
        {
            "scenario": "web_on_effort_high_no_base_url",
            "model": "openai/gpt-5.6-luna",
            "generation_config": {"reasoning_effort": "high"},
            "policy": {"web_search": True, "web_fetch": True, "mcp_only": False},
            "base_url": None,
            "api_base_url": "",
            "grader": False,
            "row": {"id": "web-1"},
        },
        {
            "scenario": "web_on_effort_high_summary_detailed",
            "model": "openai/gpt-6-sol",
            "generation_config": {"reasoning_effort": "high", "reasoning_summary": "detailed"},
            "policy": {"web_search": True, "web_fetch": True, "mcp_only": False},
            "base_url": None,
            "api_base_url": "http://127.0.0.1:13131/v1",
            "grader": False,
            "row": {"id": "web-2"},
        },
        {
            "scenario": "mcp_only_chess_effort_high",
            "model": "openai/gpt-5.6-luna",
            "generation_config": {"reasoning_effort": "high"},
            "policy": {"web_search": False, "web_fetch": False, "mcp_only": True},
            "base_url": None,
            "api_base_url": "http://127.0.0.1:13131/v1",
            "grader": True,
            "row": {"id": "chess-1", "tool_surface": "chess"},
        },
        {
            "scenario": "mcp_only_chess_effort_high_summary_detailed",
            "model": "openai/gpt-6.1-sol",
            "generation_config": {"reasoning_effort": "high", "reasoning_summary": "detailed"},
            "policy": {"web_search": False, "web_fetch": False, "mcp_only": True},
            "base_url": None,
            "api_base_url": "http://127.0.0.1:13131/v1",
            "grader": True,
            "row": {"id": "chess-2", "tool_surface": "chess"},
        },
        {
            "scenario": "custom_base_url_slug",
            "model": "openai/gpt-6-astra",
            "generation_config": {"reasoning_effort": "high", "reasoning_summary": "detailed"},
            "policy": {"web_search": False, "web_fetch": False, "mcp_only": False},
            "base_url": "https://openrouter.ai/api/v1",
            "api_base_url": None,
            "grader": True,
            "row": {"id": "or-1"},
        },
    ]

    invalid_scenarios: list[dict[str, Any]] = [
        {
            "scenario": "web_search_only",
            "model": "openai/gpt-5.6-luna",
            "generation_config": {"reasoning_effort": "high"},
            "policy": {"web_search": True, "web_fetch": False, "mcp_only": False},
        },
        {
            "scenario": "web_fetch_only",
            "model": "openai/gpt-5.6-luna",
            "generation_config": {"reasoning_effort": "high"},
            "policy": {"web_search": False, "web_fetch": True, "mcp_only": False},
        },
        {
            "scenario": "mcp_only_with_web",
            "model": "openai/gpt-5.6-luna",
            "generation_config": {"reasoning_effort": "high"},
            "policy": {"web_search": True, "web_fetch": True, "mcp_only": True},
        },
        {
            "scenario": "mcp_only_with_split_web",
            "model": "openai/gpt-5.6-luna",
            "generation_config": {"reasoning_effort": "high"},
            "policy": {"web_search": True, "web_fetch": False, "mcp_only": True},
        },
    ]

    valid_cases = [
        {"id": f"{version}__{item['scenario']}", "version": version, **item}
        for version in VERSIONS
        for item in scenarios
    ]
    invalid_cases = [
        {"id": f"{version}__{item['scenario']}", "version": version, **item}
        for version in VERSIONS
        for item in invalid_scenarios
    ]
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "valid_cases": valid_cases,
        "invalid_cases": invalid_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Codex launcher golden fixtures from original CheatBench."
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

    spec = _build_spec()
    env = {k: v for k, v in os.environ.items() if k not in {"OPENAI_BASE_URL", "OPENAI_API_KEY"}}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="cb-codex-launcher-fixtures-") as tmpdir:
        proc = subprocess.run(
            [sys.executable, "-B", "-c", _EVAL_SCRIPT, str(args.cheatbench.resolve())],
            input=json.dumps(spec),
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
        f"Wrote {len(generated['valid_cases'])} valid cases and "
        f"{len(generated['invalid_cases'])} invalid cases to {args.output}"
    )


if __name__ == "__main__":
    main()
