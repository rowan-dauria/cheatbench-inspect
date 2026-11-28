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
        if path.is_file() and path.suffix in {".py", ".json", ".jsonl", ".yaml"}
    }
    sandbox = Path("harnesses/resources/sandbox")
    expected.update({"eval.yaml", "resources/judge.txt", (sandbox / "Dockerfile").as_posix()})
    expected.add("geoguessr/resources/instruction.txt")
    expected.add("synthetic_puzzles/resources/instruction.txt")
    expected.add("knowledge_work/resources/instruction.txt")
    expected.add("knowledge_work/resources/workspace_readme.txt")
    expected.add("protein_design/resources/instruction.txt")
    expected.add("chess/resources/instruction.txt")
    expected.add("chess/resources/engine_signature.json")
    expected.add("go/resources/instruction.txt")
    expected.add("go/resources/engine_signature.json")
    expected.add("go/resources/probes/tournament_ogs_68506069.json")
    expected.add("go/resources/probes/tournament_ogs_63091411.json")
    expected.update(
        f"openmath/resources/{name}"
        for name in ("instruction.txt", "inject.txt", "mirror.txt", "materials.tar.gz")
    )
    for filename in (
        "Dockerfile",
        "office.Dockerfile",
        "chess.Dockerfile",
        "go.Dockerfile",
        "go-engine.Dockerfile",
    ):
        recipe = sandbox / filename
        expected.add(recipe.as_posix())
        for name in re.findall(
            r"--mount=type=bind,source=([^, ]+)", (package / recipe).read_text()
        ):
            if "${TARGETARCH}" in name:
                for arch in ("amd64", "arm64"):
                    expected.add((sandbox / name.replace("${TARGETARCH}", arch)).as_posix())
            else:
                expected.add((sandbox / name).as_posix())
    with ZipFile(wheel) as archive:
        shipped = set(archive.namelist())
    assert {f"cheatbench_inspect/{path}" for path in expected} <= shipped
    assert "cheatbench_inspect/prime_factorization/resources/http_verifier.py" in shipped
    assert "cheatbench_inspect/harnesses/resources/sandbox/codex-compose.yaml" in shipped
