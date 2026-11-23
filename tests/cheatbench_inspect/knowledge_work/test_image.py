import hashlib
from pathlib import Path

import pytest
import yaml

from cheatbench_inspect.harnesses.container import SANDBOX_RESOURCES


@pytest.mark.parametrize("runtime", ["terminus", "claude", "codex"])
def test_office_image_tag(runtime: str) -> None:
    inputs = ("office.Dockerfile", "office-requirements.txt")
    tag = hashlib.sha256(
        b"".join((SANDBOX_RESOURCES / name).read_bytes() for name in inputs)
    ).hexdigest()[:12]
    service = yaml.safe_load((SANDBOX_RESOURCES / f"{runtime}-office-compose.yaml").read_text())[
        "services"
    ]["default"]
    assert service["image"] == f"ghcr.io/rowan-dauria/task-env-{runtime}-office:{tag}"
    assert service["build"] == {
        "context": ".",
        "dockerfile": "office.Dockerfile",
        "target": "office",
        "args": {"RUNTIME": runtime},
    }
    assert service["network_mode"] == "none"
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/sandbox-image.yml").read_text())
    assert {
        "target": f"{runtime}-office",
        "stage": "office",
        "runtime": runtime,
    } in workflow["jobs"]["publish"]["strategy"]["matrix"]["include"]
