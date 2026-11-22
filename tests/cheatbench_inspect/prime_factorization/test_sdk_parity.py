"""Compare the SDK adapter with the original grader inside the Claude image."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SANDBOX = ROOT / "src/cheatbench_inspect/resources/sandbox"
ORIGINAL = Path("/home/agent/cheatbench/tasks/prime_factorization/grader.py")


@pytest.mark.docker
@pytest.mark.slow
def test_sdk_parity(tmp_path: Path) -> None:
    if not ORIGINAL.is_file():
        pytest.skip("original CheatBench checkout is unavailable")
    (tmp_path / "original.py").write_bytes(ORIGINAL.read_bytes())
    source = ROOT / "src/cheatbench_inspect/resources/prime_factorization"
    for name in ("verifier.py", "sdk_verifier.py"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    (tmp_path / "__init__.py").write_text("")
    script = """
import asyncio, importlib.util, sys
sys.path.insert(0, "/")
import data.original as original
import data.sdk_verifier as port
import claude_agent_sdk as sdk
captured = []
create = sdk.create_sdk_mcp_server
def capture(*args, **kwargs):
    captured.append((kwargs["name"], kwargs["version"], kwargs["tools"][0]))
    return create(*args, **kwargs)
sdk.create_sdk_mcp_server = capture
port.create_sdk_mcp_server = capture
row = {"number": "15", "id": "example"}
a = original.make_grader(row, {}, "/tmp")
b = port.make_grader(row, {}, "/tmp")
assert a[1] == b[1]
x, y = captured
assert x[:2] == y[:2]
for key in ("name", "description", "input_schema"):
    assert getattr(x[2], key) == getattr(y[2], key)
async def main():
    calls = [{"p":"2", "q":"3"}, {"p":"3", "q":"5"}, {},
             {"p":"x", "q":"2"}, {"p":"2.0", "q":"3"},
             {"p":"1", "q":"3"}, {"p":"2", "q":"3"}]
    calls += [{"p":"2", "q":"3"}] * 51
    for args in calls:
        assert await x[2].handler(args) == await y[2].handler(args)
    assert a[2] == b[2]
    c = original.make_grader(row, {}, "/tmp")
    d = port.make_grader(row, {}, "/tmp")
    t1, t2 = captured[-2][2], captured[-1][2]
    args = [{"p":"2", "q":"3"}] * 60
    r1 = await asyncio.gather(*(asyncio.to_thread(asyncio.run, t1.handler(v)) for v in args))
    r2 = await asyncio.gather(*(asyncio.to_thread(asyncio.run, t2.handler(v)) for v in args))
    assert sorted(str(v) for v in r1) == sorted(str(v) for v in r2)
    assert c[2] == d[2]
asyncio.run(main())
"""
    (tmp_path / "run.py").write_text(script)
    subprocess.run(
        ["docker", "build", "--target", "claude", "-t", "local-task-env-claude-test", str(SANDBOX)],
        check=True,
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{tmp_path}:/data:ro",
            "local-task-env-claude-test",
            "python",
            "/data/run.py",
        ],
        check=True,
    )
