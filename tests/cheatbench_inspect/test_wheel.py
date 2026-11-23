"""Ensure the installed wheel contains the runtime and image build resources."""

import re
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.mark.slow
def test_wheel_contains_runtime_resources(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "src/cheatbench_inspect"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    [wheel] = tmp_path.glob("*.whl")
    # Python resources are namespace-package modules; include them all, along
    # with the dataset and judge prompt, even when imported instead of read.
    expected = {
        path.relative_to(package).as_posix()
        for resources in package.rglob("resources")
        for path in resources.rglob("*")
        if path.is_file() and path.suffix in {".py", ".jsonl", ".yaml"}
    }
    sandbox = Path("harnesses/resources/sandbox")
    expected.update({"eval.yaml", "resources/judge.txt", (sandbox / "Dockerfile").as_posix()})
    dockerfile = (package / sandbox / "Dockerfile").read_text()
    expected.update(
        (sandbox / name).as_posix()
        for name in re.findall(r"--mount=type=bind,source=([^, ]+)", dockerfile)
    )
    with ZipFile(wheel) as archive:
        shipped = set(archive.namelist())
    assert {f"cheatbench_inspect/{path}" for path in expected} <= shipped
    assert "cheatbench_inspect/prime_factorization/resources/http_verifier.py" in shipped
    assert "cheatbench_inspect/harnesses/resources/sandbox/codex-compose.yaml" in shipped
