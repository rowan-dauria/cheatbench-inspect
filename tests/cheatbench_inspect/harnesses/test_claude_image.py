"""Verify the Claude sandbox image and its published tag."""

import hashlib
import re
import subprocess

import pytest
import yaml

from cheatbench_inspect.harnesses.container import SANDBOX_RESOURCES


def test_tag() -> None:
    inputs = ("Dockerfile", "task-requirements.txt", "claude-requirements.txt")
    tag = hashlib.sha256(
        b"".join((SANDBOX_RESOURCES / name).read_bytes() for name in inputs)
    ).hexdigest()[:12]
    compose = SANDBOX_RESOURCES / "claude-compose.yaml"
    service = yaml.safe_load(compose.read_text())["services"]["default"]
    assert compose.name.endswith("compose.yaml")
    assert service["image"] == f"ghcr.io/rowan-dauria/task-env-claude:{tag}"
    assert service["build"] == {"context": ".", "target": "claude"}
    assert "platform" not in service
    assert "extends" not in service


@pytest.mark.docker
@pytest.mark.slow
def test_image() -> None:
    image = "local-task-env-claude-test"
    subprocess.run(
        ["docker", "build", "--target", "claude", "-t", image, str(SANDBOX_RESOURCES)], check=True
    )

    task_image = "local-task-env-base-test"
    subprocess.run(
        ["docker", "build", "--target", "task", "-t", task_image, str(SANDBOX_RESOURCES)],
        check=True,
    )
    packages = ["dpkg-query", "-W", "-f=${binary:Package} ${Version} ${db:Status-Abbrev}\n"]
    task_packages = subprocess.check_output(
        ["docker", "run", "--rm", task_image, *packages], text=True
    )
    claude_packages = subprocess.check_output(
        ["docker", "run", "--rm", image, *packages], text=True
    )
    assert task_packages == claude_packages

    def run(*args: str) -> str:
        return subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                "1000:1000",
                "-e",
                "HOME=/home/user",
                image,
                *args,
            ],
            text=True,
        ).strip()

    assert (
        run(
            "python",
            "-c",
            "import claude_agent_sdk, anthropic; print(claude_agent_sdk.__version__, anthropic.__version__)",
        )
        == "0.2.135 0.121.0"
    )
    assert run("node", "--version") == "v22.23.5"
    assert run("npm", "--version") == "10.9.9"
    assert (
        run("readlink", "-f", "/usr/local/bin/node")
        == "/home/user/.nvm/versions/node/v22.23.5/bin/node"
    )
    run("test", "-w", "/home/user/.nvm/versions/node/v22.23.5/bin")
    expected = set()
    for name in ("task-requirements.txt", "claude-requirements.txt"):
        expected.update(
            re.findall(r"(?m)^([a-zA-Z0-9_.-]+==[^\s;\\]+)", (SANDBOX_RESOURCES / name).read_text())
        )
    assert {s.lower().replace("_", "-") for s in run("pip", "freeze").splitlines()} == {
        s.lower().replace("_", "-") for s in expected
    }
    run(
        "sh",
        "-c",
        "! grep -rilsIE --exclude-dir=site-packages --exclude-dir=.nvm --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev --exclude-dir=lib --exclude-dir=share 'cheatbench|claude-harness' / 2>/dev/null",
    )
