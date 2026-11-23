"""The Terminus compose file names the published image its build inputs produce."""

import hashlib

import yaml

from cheatbench_inspect.harnesses.terminus import TERMINUS_COMPOSE

IMAGE = "ghcr.io/rowan-dauria/task-env-terminus"
BUILD_INPUTS = ("Dockerfile", "task-requirements.txt")


def test_compose_names_the_image_tag_of_its_build_inputs() -> None:
    # Mirrors .github/workflows/sandbox-image.yml, which publishes this tag.
    sandbox = TERMINUS_COMPOSE.parent
    inputs = b"".join((sandbox / name).read_bytes() for name in BUILD_INPUTS)
    tag = hashlib.sha256(inputs).hexdigest()[:12]

    service = yaml.safe_load(TERMINUS_COMPOSE.read_text(encoding="utf-8"))["services"]["default"]

    assert service["image"] == f"{IMAGE}:{tag}"
    assert service["build"] == {"context": ".", "target": "terminus"}


def test_compose_is_flat_for_hawk() -> None:
    # Hawk's compose-to-Kubernetes converter rejects service keys it does not map,
    # including `extends`.
    service = yaml.safe_load(TERMINUS_COMPOSE.read_text(encoding="utf-8"))["services"]["default"]

    assert "extends" not in service


def test_compose_file_name_marks_it_as_compose_for_hawk() -> None:
    # Hawk converts a sandbox file as compose only when its name ends in compose.yaml
    # (k8s_sandbox.compose.is_docker_compose_file). Any other name is read as Helm
    # values, which Hawk refuses for compose keys such as `build` and `init`.
    assert TERMINUS_COMPOSE.name.endswith("compose.yaml")
