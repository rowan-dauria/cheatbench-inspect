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
    resources = package / "resources"
    expected = {
        path.relative_to(package).as_posix()
        for path in resources.rglob("*")
        if path.is_file() and path.suffix in {".py", ".jsonl", ".yaml"}
    }
    expected.update({"eval.yaml", "resources/judge.txt", "resources/sandbox/Dockerfile"})
    dockerfile = (resources / "sandbox/Dockerfile").read_text()
    expected.update(
        f"resources/sandbox/{name}"
        for name in re.findall(r"--mount=type=bind,source=([^, ]+)", dockerfile)
    )
    with ZipFile(wheel) as archive:
        shipped = set(archive.namelist())
    assert {f"cheatbench_inspect/{path}" for path in expected} <= shipped
    assert "cheatbench_inspect/resources/codex/episode.py" in shipped
    assert "cheatbench_inspect/resources/codex/proxy.py" in shipped
    assert "cheatbench_inspect/resources/prime_factorization/http_verifier.py" in shipped
