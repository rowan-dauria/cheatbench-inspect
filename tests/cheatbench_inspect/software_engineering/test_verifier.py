"""Unit tests for correctness verifier classification and suite-load vs assertion errors."""

from __future__ import annotations

from pathlib import Path

from cheatbench_inspect.software_engineering.verifier import (
    classify_verifier_result,
    parse_test_list,
)


def test_parse_test_list_handles_json_literal_and_list() -> None:
    assert parse_test_list(["a", "b"]) == ["a", "b"]
    assert parse_test_list('["a", "b"]') == ["a", "b"]
    assert parse_test_list("['a', 'b']") == ["a", "b"]
    assert parse_test_list("") == []
    assert parse_test_list(None) == []
    assert parse_test_list("not valid") == []


def test_classify_resolved_episode() -> None:
    config = {"fail_to_pass": '["t1"]', "pass_to_pass": '["t2"]'}
    output = {"tests": [{"name": "t1", "status": "PASSED"}, {"name": "t2", "status": "PASSED"}]}
    report = classify_verifier_result(
        config=config,
        output_json=output,
        exit_code=0,
        reward_text="1\n",
        stdout="ok",
        stderr="",
    )
    assert report["resolved"] is True
    assert report["reward"] == 1
    assert report["suite_load_error"] is False
    assert report["assertion_failure"] is False
    assert report["fail_to_pass"]["passed"] == 1
    assert report["pass_to_pass"]["passed"] == 1


def test_classify_assertion_failure_separately_from_suite_load_error() -> None:
    # Ansible baseline pattern: test ran and failed an assertion.
    config = {
        "fail_to_pass": ["test_get_sysinfo_facts.py::test_get_sysinfo_facts"],
        "pass_to_pass": [],
    }
    output = {
        "tests": [{"name": "test_get_sysinfo_facts.py::test_get_sysinfo_facts", "status": "FAILED"}]
    }
    report = classify_verifier_result(
        config=config,
        output_json=output,
        exit_code=1,
        reward_text="0\n",
        stdout="AttributeError: 'LinuxHardware' object has no attribute 'get_sysinfo_facts'",
        stderr="",
    )
    assert report["resolved"] is False
    assert report["reward"] == 0
    assert report["assertion_failure"] is True
    assert report["suite_load_error"] is False
    assert report["fail_to_pass"]["failed"] == 1
    assert report["fail_to_pass"]["missing"] == 0


def test_classify_protonmail_style_suite_load_error() -> None:
    # ProtonMail baseline pattern: test runner failed during suite load -> empty output.json.
    config = {"fail_to_pass": ["test_a", "test_b"], "pass_to_pass": []}
    report = classify_verifier_result(
        config=config,
        output_json={"tests": []},
        exit_code=1,
        reward_text="0\n",
        stdout="",
        stderr="TS2305: Module has no exported member",
    )
    assert report["resolved"] is False
    assert report["reward"] == 0
    assert report["suite_load_error"] is True
    assert report["assertion_failure"] is False
    assert report["fail_to_pass"]["missing"] == 2
    assert report["total_parsed_tests"] == 0


def test_classify_collection_error_and_verifier_timeout() -> None:
    config = {"fail_to_pass": ["test_a"], "pass_to_pass": []}
    collection = classify_verifier_result(
        config=config,
        output_json={"tests": [{"name": "test_a", "status": "ERROR"}]},
        exit_code=1,
        reward_text="0",
        stdout="",
        stderr="ImportError",
    )
    assert collection["suite_load_error"] is True
    assert collection["assertion_failure"] is False

    timeout = classify_verifier_result(
        config=config,
        output_json=None,
        exit_code=None,
        reward_text=None,
        stdout="",
        stderr="",
        verifier_error="Verifier timed out after 6000s",
    )
    assert timeout["resolved"] is False
    assert timeout["suite_load_error"] is True
    assert timeout["verifier_error"] == "Verifier timed out after 6000s"


def _make_tests_dir(tmp_path: Path, instance_id: str) -> Path:
    tests_dir = tmp_path / "instances" / instance_id / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (tests_dir / "run_script.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (tests_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")
    (tests_dir / "config.json").write_text(
        '{"fail_to_pass": ["t1"], "pass_to_pass": []}', encoding="utf-8"
    )
    return tmp_path


async def test_run_correctness_verifier_mocked_sandbox_paths(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    from inspect_ai.util import ExecResult

    from cheatbench_inspect.software_engineering import verifier as verifier_mod

    task_dir = _make_tests_dir(tmp_path, "inst-1")

    # 1. Prep failure (rm/mkdir fails) returns suite_load_error with verifier_error.
    box = MagicMock()
    box.exec = AsyncMock(return_value=ExecResult(False, 1, "", "disk full"))
    box.write_file = AsyncMock()
    with patch.object(verifier_mod, "sandbox", return_value=box):
        report = await verifier_mod.run_correctness_verifier(task_dir, "inst-1")
    assert report["suite_load_error"] is True
    assert "Could not create /tests" in str(report["verifier_error"])
    # Verify prep clears stale /logs/verifier and /tmp/output.json before mkdir.
    assert "rm -rf /logs/verifier /tmp/output.json" in box.exec.await_args.args[0][2]

    # 2. Timeout during bash /tests/test.sh returns verifier_error cleanly.
    box = MagicMock()
    box.exec = AsyncMock(
        side_effect=[
            ExecResult(True, 0, "", ""),  # prep
            ExecResult(True, 0, "", ""),  # chmod
            ExecResult(True, 0, "", ""),  # pycache cleanup
            TimeoutError(),  # test.sh timeout
        ]
    )
    box.write_file = AsyncMock()
    with patch.object(verifier_mod, "sandbox", return_value=box):
        report = await verifier_mod.run_correctness_verifier(task_dir, "inst-1", timeout_s=42)
    assert report["suite_load_error"] is True
    assert report["verifier_error"] == "Verifier timed out after 42s"

    # 3. Missing /logs/verifier/output.json falls back to /tmp/output.json and handles malformed JSON.
    box = MagicMock()
    box.exec = AsyncMock(return_value=ExecResult(True, 0, "ok", ""))
    box.write_file = AsyncMock()
    box.read_file = AsyncMock(side_effect=[FileNotFoundError(), "{not-json", "0\n", "out", "err"])
    with patch.object(verifier_mod, "sandbox", return_value=box):
        report = await verifier_mod.run_correctness_verifier(task_dir, "inst-1")
    assert report["suite_load_error"] is True
    assert "Invalid verifier output.json" in str(report["verifier_error"])

    # 4. Neither /logs/verifier/output.json nor /tmp/output.json exists.
    box = MagicMock()
    box.exec = AsyncMock(
        side_effect=[
            ExecResult(True, 0, "", ""),  # prep
            ExecResult(True, 0, "", ""),  # chmod
            ExecResult(True, 0, "", ""),  # pycache cleanup
            ExecResult(False, 1, "raw out", "raw err"),  # test.sh
        ]
    )
    box.write_file = AsyncMock()
    box.read_file = AsyncMock(side_effect=FileNotFoundError())
    with patch.object(verifier_mod, "sandbox", return_value=box):
        report = await verifier_mod.run_correctness_verifier(task_dir, "inst-1")
    assert report["suite_load_error"] is True
    assert report["verifier_error"] == "Verifier did not produce /logs/verifier/output.json"
    assert report["stdout_tail"] == "raw out"
    assert report["stderr_tail"] == "raw err"
