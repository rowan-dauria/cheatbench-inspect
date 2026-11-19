from importlib.resources import files

import yaml

from cheatbench_inspect import cheatbench_sycophancy


def test_sycophancy_metadata_matches_task() -> None:
    metadata = yaml.safe_load(
        files("cheatbench_inspect").joinpath("eval.yaml").read_text(encoding="utf-8")
    )
    declared_tasks = {entry["name"]: entry for entry in metadata["tasks"]}

    task = cheatbench_sycophancy()

    assert "cheatbench_sycophancy" in declared_tasks
    assert declared_tasks["cheatbench_sycophancy"]["dataset_samples"] == len(task.dataset) == 28
    assert task.version == metadata["version"]
