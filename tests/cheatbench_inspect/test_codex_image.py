"""Verify the Codex image build inputs and workflow mapping."""

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = ROOT / "src/cheatbench_inspect/resources/sandbox"


def test_codex_tag() -> None:
    inputs = ("Dockerfile", "task-requirements.txt")
    tag = hashlib.sha256(b"".join((SANDBOX / name).read_bytes() for name in inputs)).hexdigest()[
        :12
    ]
    service = yaml.safe_load((SANDBOX / "codex-compose.yaml").read_text())["services"]["default"]
    assert service["image"] == f"ghcr.io/rowan-dauria/task-env-codex:{tag}"
    assert service["build"] == {"context": ".", "target": "task"}
    workflow = yaml.safe_load((ROOT / ".github/workflows/sandbox-image.yml").read_text())
    matrix = workflow["jobs"]["publish"]["strategy"]["matrix"]["include"]
    assert {"target": "codex", "stage": "task"} in matrix
