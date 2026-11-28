import hashlib
import json
import runpy
import tarfile
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

import pytest
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver
from tests.cheatbench_inspect.openmath.conftest import (
    API,
    ARCHIVES,
    BACKGROUND,
    BLOG,
    FIXTURES,
    POSTS,
    POSTS_API,
    ROWS,
    TITLES,
)

from cheatbench_inspect import cheatbench_openmath
from cheatbench_inspect.openmath.build import SITE_BUNDLE
from cheatbench_inspect.openmath.dataset import default_task_dir, row_responses
from cheatbench_inspect.openmath.openmath import openmath_solver
from cheatbench_inspect.openmath.scorers import openmath_scorer


@pytest.mark.parametrize("harness", ["terminus-2", "kimi-code"])
def test_task_and_prompt(task_dir: Path, harness: str) -> None:
    version = "0.43.0" if harness == "kimi-code" else None
    model = "moonshotai/kimi-k3" if harness == "kimi-code" else None
    task = cheatbench_openmath(
        task_dir=str(task_dir), harness=harness, harness_version=version, harness_model=model
    )
    assert task.sandbox is not None
    sandbox_stem = "base" if harness == "kimi-code" else "terminus"
    assert Path(str(task.sandbox.config)).name == f"{sandbox_stem}-bridge-compose.yaml"
    assert len(task.dataset) == 2
    sample = task.dataset[0]
    assert sample.id == ROWS[0]
    assert sample.input == (FIXTURES / "prompt.txt").read_text()
    assert not sample.files
    original_row = json.loads((FIXTURES / "rows.jsonl").read_text().splitlines()[0])
    assert "deliverable_files" not in original_row
    assert task.version == "6-A"
    assert task.message_limit == 300
    assert task.time_limit is None
    assert task.epochs == 1
    assert isinstance(openmath_solver(str(task_dir)), Solver)
    assert isinstance(openmath_scorer(), Scorer)


def build_mirror(task_dir: Path, tmp_path: Path, module: Path, row: str):
    directory = tmp_path / row
    capture = directory / "capture"
    with tarfile.open(task_dir / SITE_BUNDLE) as archive:
        archive.extractall(capture, filter="data")
    for name, contents in row_responses(task_dir, row).items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    mirror = runpy.run_path(str(module))["Mirror"]
    return mirror(capture, directory / "overrides.json", only=row)


@pytest.mark.parametrize("index", [0, 1])
def test_routing_is_row_isolated(
    task_dir: Path, tmp_path: Path, mirror_code: Path, index: int
) -> None:
    row, sibling = ROWS[index], 1 - index
    mirror = build_mirror(task_dir, tmp_path, mirror_code, row)
    status, _, body = mirror.handle(BLOG, BACKGROUND)
    assert status == 200
    assert TITLES[index].encode() in body
    assert TITLES[sibling].encode() not in body
    assert mirror.handle(BLOG, POSTS[index])[0] == 200
    assert mirror.handle(BLOG, POSTS[index].rstrip("/"))[0] == 301
    # Upstream applies its hostless overrides before branching to the API host.
    for host in (BLOG, API):
        status, _, body = mirror.handle(host, ARCHIVES[index])
        assert status == 200
        assert body == (task_dir / f"files/archives/{row}.zip").read_bytes()
        assert mirror.handle(host, ARCHIVES[sibling])[0] == 404
    for path in (POSTS[sibling], POSTS[sibling].rstrip("/")):
        assert mirror.handle(BLOG, path)[0] == 404
    status, _, body = mirror.handle(API, POSTS_API)
    assert status == 200
    assert [post["URL"] for post in json.loads(body)["posts"]] == [
        f"https://{BLOG}{POSTS[index]}",
        f"https://{BLOG}{BACKGROUND}",
    ]


@pytest.mark.dataset_download
@pytest.mark.slow
def test_release_matches_upstream_mirror(tmp_path: Path, mirror_code: Path) -> None:
    task_dir = default_task_dir(tmp_path / "cache")
    assert default_task_dir(tmp_path / "cache") == task_dir
    task = cheatbench_openmath(task_dir=str(task_dir))
    assert len(task.dataset) == 20
    assert len({sample.id for sample in task.dataset}) == 20
    for sample in task.dataset:
        with ZipFile(task_dir / f"files/archives/{sample.id}.zip") as archive:
            assert archive.namelist() == ["writeup.md", "certificate.txt", "verify.py"]
            assert [member.external_attr >> 16 for member in archive.infolist()] == [0o100644] * 3
    expected = json.loads((FIXTURES / "responses.json").read_text())
    for row, request in expected.items():
        mirror = build_mirror(task_dir, tmp_path, mirror_code, row)
        for url, result in request.items():
            parts = urlsplit(url)
            assert parts.hostname is not None
            status, _, body = mirror.handle(parts.hostname, parts.path)
            assert [status, hashlib.sha256(body).hexdigest()] == result, (row, url)
