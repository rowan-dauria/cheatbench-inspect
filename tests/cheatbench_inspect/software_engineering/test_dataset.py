"""Dataset manifest, prompt derivation, SHA-256 verification and per-sample sandbox tests."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from cheatbench_inspect.software_engineering.dataset import (
    TEST_FILENAMES,
    WORKTEST_LINE,
    build_task_dir,
    compose_file_for_instance,
    default_task_dir,
    download_instance_solution,
    get_dataset,
    load_manifest,
    worktest_instruction,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DIGEST_IMAGE_RE = re.compile(r"^jefzda/sweap-images:[^@]+@sha256:[0-9a-f]{64}$")


def test_manifest_pins_all_20_instances_by_digest_and_sha256() -> None:
    manifest = load_manifest()
    assert manifest["cheatbench_commit"] == "96e05a6d61faf33be788f75dd01998066045ad02"
    assert manifest["harbor_datasets_repo"] == "laude-institute/harbor-datasets"
    assert manifest["harbor_datasets_commit"] == "c8e8f3fac7097accaacf261d74c3d6f441de45b1"

    instances = manifest["instances"]
    assert len(instances) == 20
    assert len({item["id"] for item in instances}) == 20
    assert len({item["repo"] for item in instances}) == 11

    linux_header_ids = [item["id"] for item in instances if item["install_linux_headers"]]
    assert len(linux_header_ids) == 2
    assert all("gravitational__teleport" in item_id for item_id in linux_header_ids)

    extra_blank_ids = [item["id"] for item in instances if item["extra_trailing_blank_line"]]
    assert extra_blank_ids == [
        "instance_qutebrowser__qutebrowser-70248f256f93ed9b1984494d0a1a919ddd774892-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
    ]

    for item in instances:
        assert DIGEST_IMAGE_RE.match(item["image"]), item["image"]
        assert re.fullmatch(r"[0-9a-f]{40}", item["base_commit"])
        assert SHA256_RE.match(item["upstream_instruction_sha256"])
        for relpath in (
            "instruction.md",
            *(f"tests/{f}" for f in TEST_FILENAMES),
            "solution/solve.sh",
        ):
            assert SHA256_RE.match(item["sha256"][relpath]), (item["id"], relpath)


def test_worktest_instruction_extraction_and_override() -> None:
    upstream = "Preamble\n<pr_description>\nFix bug X.\n\n\n</pr_description>\nFooter\n"
    assert worktest_instruction(upstream) == f"Fix bug X.\n\n\n{WORKTEST_LINE}\n"
    assert (
        worktest_instruction(upstream, extra_trailing_blank_line=True)
        == f"Fix bug X.\n\n\n\n{WORKTEST_LINE}\n"
    )
    with pytest.raises(ValueError, match="no <pr_description> block"):
        worktest_instruction("No tags here.")


def test_build_task_dir_verifies_checksums_and_builds_anonymous_compose(tmp_path: Path) -> None:
    manifest = load_manifest()
    first = manifest["instances"][0]
    raw_instruction = b"Header\n<pr_description>\nDo the task.\n</pr_description>\n"
    derived_instruction = worktest_instruction(raw_instruction.decode("utf-8")).encode("utf-8")
    fake_files = {
        "instruction.md": raw_instruction,
        "tests/test.sh": b"#!/bin/bash\nexit 0\n",
        "tests/run_script.sh": b"#!/bin/bash\nexit 0\n",
        "tests/parser.py": b"print('ok')\n",
        "tests/config.json": b'{"fail_to_pass": [], "pass_to_pass": []}\n',
        "solution/solve.sh": b"#!/bin/bash\nexit 0\n",
    }
    patched_instance = {
        **first,
        "upstream_instruction_sha256": hashlib.sha256(raw_instruction).hexdigest(),
        "sha256": {
            "instruction.md": hashlib.sha256(derived_instruction).hexdigest(),
            **{
                k: hashlib.sha256(v).hexdigest()
                for k, v in fake_files.items()
                if k != "instruction.md"
            },
        },
    }
    patched_manifest = {**manifest, "instances": [patched_instance]}

    def fake_download(url: str) -> bytes:
        for relpath, content in fake_files.items():
            if url.endswith(f"/{relpath}"):
                return content
        raise AssertionError(f"Unexpected URL: {url}")

    from unittest.mock import patch

    out = tmp_path / "built"
    with patch(
        "cheatbench_inspect.software_engineering.dataset.load_manifest",
        return_value=patched_manifest,
    ):
        build_task_dir(out, download=fake_download)
        solution = download_instance_solution(first["id"], download=fake_download)
        assert solution == fake_files["solution/solve.sh"]

        # Corrupt bytes fail loudly.
        with pytest.raises(ValueError, match="has sha256"):
            build_task_dir(tmp_path / "bad", download=lambda _url: b"corrupted")

        with pytest.raises(ValueError, match="has sha256"):
            download_instance_solution(first["id"], download=lambda _url: b"corrupted")

        with pytest.raises(ValueError, match="Unknown SWE-bench Pro instance ID"):
            download_instance_solution("missing-id", download=fake_download)

    dataset = get_dataset(out, compose_dir=tmp_path / "compose")
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.id == first["id"]
    assert sample.input == derived_instruction.decode("utf-8")
    assert sample.files == {}
    assert sample.sandbox is not None
    assert sample.sandbox.type == "docker"
    assert isinstance(sample.sandbox.config, str)
    compose_path = Path(sample.sandbox.config)
    assert compose_path.name.endswith("-compose.yaml")
    compose_text = compose_path.read_text(encoding="utf-8")
    assert "cheatbench" not in compose_path.name.lower()
    assert "cheatbench" not in compose_text.lower()
    parsed = yaml.safe_load(compose_text)
    service = parsed["services"]["default"]
    assert service["image"] == first["image"]
    assert service["platform"] == "linux/amd64"
    assert service["cpus"] == 8
    assert service["mem_limit"] == "16384m"
    assert service["working_dir"] == "/app"
    assert service["network_mode"] == "bridge"
    assert service["environment"]["PYTHONPATH"] == "/app/lib:/app"


def test_compose_file_for_instance_is_idempotent(tmp_path: Path) -> None:
    record = {"id": "inst-1", "image": "jefzda/sweap-images:tag@sha256:" + "0" * 64}
    first = compose_file_for_instance(record, tmp_path)
    second = compose_file_for_instance(record, tmp_path)
    assert first == second


def test_record_to_sample_with_real_manifest_record_and_unknown_id_error(tmp_path: Path) -> None:
    from cheatbench_inspect.software_engineering.dataset import record_to_sample

    manifest = load_manifest()
    ansible = next(item for item in manifest["instances"] if "cd9c4eb5" in item["id"])
    inst_dir = tmp_path / "instances" / ansible["id"]
    inst_dir.mkdir(parents=True)
    real_prompt = f'"# Changes to linux.py for setup module"\n\n\n{WORKTEST_LINE}\n'
    (inst_dir / "instruction.md").write_text(real_prompt, encoding="utf-8")
    record = {
        "id": ansible["id"],
        "repo": ansible["repo"],
        "repo_language": ansible["repo_language"],
        "base_commit": ansible["base_commit"],
        "image": ansible["image"],
        "install_linux_headers": ansible["install_linux_headers"],
        "instruction_sha256": ansible["sha256"]["instruction.md"],
        "instruction_path": f"instances/{ansible['id']}/instruction.md",
        "tests_dir": f"instances/{ansible['id']}/tests",
    }
    sample = record_to_sample(record, task_dir=tmp_path, compose_dir=tmp_path / "compose")
    assert sample.id == ansible["id"]
    assert sample.input == real_prompt
    assert sample.target == ""
    assert sample.metadata is not None
    assert sample.metadata["repo"] == "ansible/ansible"
    assert sample.metadata["base_commit"] == "585ef6c55e87c10c1ce7d59ebe9c33dd6dbe5afb"
    assert sample.metadata["install_linux_headers"] is False
    assert sample.sandbox is not None
    assert sample.sandbox.type == "docker"

    with pytest.raises(ValueError, match="Unknown SWE-bench Pro instance IDs"):
        build_task_dir(tmp_path / "missing", instance_ids=["does-not-exist"])


@pytest.mark.dataset_download
@pytest.mark.slow
def test_default_task_dir_downloads_all_20_verified_instances(tmp_path: Path) -> None:
    task_dir = default_task_dir(tmp_path / "cache")
    assert default_task_dir(tmp_path / "cache") == task_dir
    rows = [
        json.loads(line)
        for line in (task_dir / "data.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 20
    dataset = get_dataset(task_dir)
    assert len(dataset) == 20
    for sample in dataset:
        assert isinstance(sample.input, str)
        assert sample.input.endswith(f"{WORKTEST_LINE}\n")
