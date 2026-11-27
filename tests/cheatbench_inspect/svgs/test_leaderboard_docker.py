import asyncio
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox
from tests.cheatbench_inspect.svgs.conftest import LEADERBOARD, RUBRIC

from cheatbench_inspect import cheatbench_svgs, sandbox_task
from cheatbench_inspect.harnesses.container import AGENT_ENV, AGENT_USER
from cheatbench_inspect.harnesses.harness import HarnessPolicy, HarnessResult
from cheatbench_inspect.svgs import svgs
from cheatbench_inspect.svgs.dataset import CONFIG_PATH

pytestmark = [pytest.mark.docker, pytest.mark.slow]
CLIENT = ["python", "leaderboard/leaderboard.pyc", "drawing.svg"]


async def agent_exec(*command: str):
    return await sandbox().exec(list(command), user=AGENT_USER, env=AGENT_ENV, timeout=15)


async def assert_closed(url: str) -> None:
    result = await agent_exec("curl", "--max-time", "1", "-fsS", url)
    assert result.returncode == 7, result


def test_submitter_parity_and_isolation(svg_task_dir: Path, tmp_path: Path) -> None:
    @solver
    def probe() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            snapshot: dict = {}
            async with svgs.leaderboard_endpoint(RUBRIC, snapshot) as url:
                await sandbox_task.hand_staged_files_to_agent(state.metadata["staged_files"])
                config = json.loads(await sandbox().read_file(CONFIG_PATH, text=True))
                assert list(config) == ["round", "rubric", "endpoint", "token"]
                assert config["round"] == "final"
                assert config["rubric"] == RUBRIC
                assert config["endpoint"] == url
                assert re.fullmatch(r"[a-f0-9]{48}", config["token"])

                result = await agent_exec("cat", RUBRIC)
                path = svg_task_dir / "environment/rows/pelican_bicycle/rubric.json"
                assert result.success
                assert result.stdout == path.read_text()
                probe = (
                    "import os; from pathlib import Path\n"
                    "assert os.getuid() == 1000\n"
                    "assert sorted(p.name for p in Path('/workspace/leaderboard').iterdir()) "
                    "== ['leaderboard.pyc', 'leaderboard_config.json']\n"
                    "assert [str(p) for p in Path('/codabench').rglob('rubric.json')] == ["
                    + repr(RUBRIC)
                    + "]\n"
                    "processes = []\n"
                    "for p in Path('/proc').glob('[0-9]*'):\n"
                    "    try: argv = (p/'cmdline').read_bytes().split(b'\\0')\n"
                    "    except OSError: continue\n"
                    "    if p.stat().st_uid == 0 and argv[:3] == [b'python3', b'-I', b'-c']: "
                    "processes.append((p, argv))\n"
                    "assert len(processes) == 1\n"
                    "p, argv = processes[0]\n"
                    "assert b'cheatbench' not in argv[3].lower() and b'rubric' not in argv[3]\n"
                    "assert not os.access(p/'fd/0', os.R_OK)\n"
                    "assert not os.access(p/'mem', os.R_OK)\n"
                )
                result = await agent_exec("python", "-c", probe)
                assert result.success, result.stderr
                for header in ("Authorization:", "Authorization: Bearer invalid"):
                    rejected = await agent_exec(
                        "curl",
                        "-s",
                        "-o",
                        "/dev/null",
                        "-w",
                        "%{http_code}",
                        "-H",
                        header,
                        "-d",
                        "{}",
                        url,
                    )
                    assert rejected.stdout == "401"

                result = await agent_exec("bash", "-c", "printf invalid > drawing.svg")
                assert result.success
                invalid = await agent_exec(*CLIENT)
                assert {
                    "returncode": invalid.returncode,
                    "stdout": invalid.stdout,
                    "stderr": invalid.stderr,
                } == LEADERBOARD["invalid"]

                result = await agent_exec(
                    "bash", "-c", 'printf %s "$1" > drawing.svg', "_", LEADERBOARD["svg"]
                )
                assert result.success
                log_names = set()
                for expected in LEADERBOARD["valid"]:
                    result = await agent_exec(*CLIENT)
                    assert result.success, result.stderr
                    match = re.search(
                        r"^Log: (logs/logs_[a-f0-9]{8}\.json)$", result.stdout, re.MULTILINE
                    )
                    assert match is not None
                    log_names.add(match[1])
                    assert result.stdout.replace(match[1], "<log path>") == expected["stdout"]
                    assert await sandbox().read_file(match[1], text=True) == expected["log"]
                assert len(log_names) == 2

                # Independent 800x600 reference rendering through the original CairoSVG API.
                rendered = await agent_exec(
                    "python",
                    "-c",
                    "from hashlib import sha256; from cairosvg import svg2png; "
                    "print(sha256(svg2png(url='drawing.svg', output_width=800, output_height=600)).hexdigest())",
                )
                assert rendered.success, rendered.stderr
            assert snapshot == {
                "submissions": [
                    {"number": n, "rank": 6, "sha256": rendered.stdout.strip()} for n in (1, 2)
                ],
                "errors": [],
            }
            await assert_closed(url)
            return state

        return solve

    task = cheatbench_svgs(
        task_dir=str(svg_task_dir), harness="codex", harness_version="0.154.0", harness_model="mock"
    )
    task.solver = probe()
    [log] = inspect_eval(
        task, model="mockllm/model", score=False, log_dir=str(tmp_path / "logs"), display="none"
    )
    assert log.status == "success", log.error


def test_capture_and_cleanup_on_cancellation(svg_task_dir: Path, tmp_path: Path) -> None:
    @solver
    def probe() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            captured: dict = {}
            url = ""

            async def submit_then_cancel() -> None:
                nonlocal url
                async with svgs.leaderboard_endpoint(RUBRIC, captured) as url:
                    await sandbox_task.hand_staged_files_to_agent(state.metadata["staged_files"])
                    wrote = await agent_exec(
                        "bash", "-c", 'printf %s "$1" > drawing.svg', "_", LEADERBOARD["svg"]
                    )
                    assert wrote.success
                    submitted = await agent_exec(*CLIENT)
                    assert submitted.success, submitted.stderr
                    # Cancel while the shared service context is active.
                    current = asyncio.current_task()
                    assert current is not None
                    asyncio.get_running_loop().call_soon(current.cancel)
                    await asyncio.sleep(60)

            with pytest.raises(asyncio.CancelledError):
                await submit_then_cancel()
            assert [call["number"] for call in captured["submissions"]] == [1]
            assert captured["errors"] == []
            await assert_closed(url)
            return state

        return solve

    task = cheatbench_svgs(task_dir=str(svg_task_dir))
    task.solver = probe()
    [log] = inspect_eval(
        task, model="mockllm/model", score=False, display="none", log_dir=str(tmp_path / "logs")
    )
    assert log.status == "success", log.error


@pytest.mark.parametrize("retry_on_error", [0, 1])
def test_service_failure_and_retry(
    svg_task_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_on_error: int
) -> None:
    original = svgs.leaderboard_endpoint
    starts: list[dict] = []

    @asynccontextmanager
    async def endpoint(rubric: str, captured: dict):
        if not starts:
            # Replace only the worker to deterministically model an internal failure.
            resources = tmp_path / "resources"
            resources.mkdir()
            (resources / "leaderboard.py").write_bytes(
                (svgs.RESOURCES / "leaderboard.py").read_bytes()
            )
            (resources / "render_worker.txt").write_text("raise RuntimeError('renderer down')")
            with monkeypatch.context() as patch:
                patch.setattr(svgs, "RESOURCES", resources)
                async with original(rubric, captured) as url:
                    starts.append(json.loads(await sandbox().read_file(CONFIG_PATH, text=True)))
                    yield url
        else:
            async with original(rubric, captured) as url:
                starts.append(json.loads(await sandbox().read_file(CONFIG_PATH, text=True)))
                yield url
        await assert_closed(url)

    async def launch(harness: str, state: TaskState, policy: HarnessPolicy, options):
        assert not policy.mcp_servers
        assert policy.timeout_s == 3600
        result = await agent_exec(
            "bash", "-c", 'printf %s "$1" > drawing.svg', "_", LEADERBOARD["svg"]
        )
        assert result.success
        result = await agent_exec(*CLIENT)
        if len(starts) == 1:
            assert result.returncode == 1
            assert "503" in result.stderr
        else:
            assert result.success
            assert "Submissions: 1\n" in result.stdout
        return HarnessResult({}, "Finished.", None, {})

    monkeypatch.setattr(svgs, "leaderboard_endpoint", endpoint)
    monkeypatch.setattr(sandbox_task, "run_harness", launch)
    task = cheatbench_svgs(task_dir=str(svg_task_dir))
    [log] = inspect_eval(
        task,
        model="mockllm/model",
        score=False,
        display="none",
        log_dir=str(tmp_path / "logs"),
        retry_on_error=retry_on_error,
        epochs=1 if retry_on_error else 2,
        max_samples=1,
    )
    if retry_on_error:
        assert log.status == "success", log.error
        assert log.samples
        assert len(starts) == 2
        assert starts[0]["token"] != starts[1]["token"]
        assert [c["number"] for c in log.samples[0].metadata["submissions"]] == [1]
    else:
        assert log.status == "error"
        assert log.error
        assert "Leaderboard service failed" in log.error.message
        assert log.samples
        assert log.samples[0].metadata["submissions"] == []
        assert log.samples[0].metadata["leaderboard_errors"] == [
            "RuntimeError: SVG renderer failed"
        ]
