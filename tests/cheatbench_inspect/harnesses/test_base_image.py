"""Verify the base image build inputs and workflow mapping."""

import hashlib
from pathlib import Path

import yaml

from cheatbench_inspect.harnesses.container import SANDBOX_RESOURCES

ROOT = Path(__file__).resolve().parents[3]


def test_base_tag() -> None:
    inputs = ("Dockerfile", "task-requirements.txt")
    tag = hashlib.sha256(
        b"".join((SANDBOX_RESOURCES / name).read_bytes() for name in inputs)
    ).hexdigest()[:12]
    service = yaml.safe_load((SANDBOX_RESOURCES / "base-compose.yaml").read_text())["services"][
        "default"
    ]
    assert service["image"] == f"ghcr.io/rowan-dauria/task-env-base:{tag}"
    assert service["build"] == {"context": ".", "target": "task"}
    workflow = yaml.safe_load((ROOT / ".github/workflows/sandbox-image.yml").read_text())
    matrix = workflow["jobs"]["publish"]["strategy"]["matrix"]["include"]
    assert {"target": "base", "stage": "task"} in matrix
