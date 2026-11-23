"""Generate judge and prompt parity fixtures from the original and the built archive.

Run with uv run python -B scripts/synthetic_puzzles_fixtures.py ARCHIVE.
"""

import argparse
import ast
import json
import runpy
import sys
import tarfile
import tempfile
import types
from pathlib import Path

UPSTREAM = Path("/home/agent/cheatbench")
DESTINATION = (
    Path(__file__).resolve().parents[1] / "tests/cheatbench_inspect/synthetic_puzzles/fixtures"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    original = UPSTREAM / "tasks/synthetic_puzzles"

    sys.path.insert(0, str(UPSTREAM))
    llm_agents = types.ModuleType("core.llm_agents")
    llm_agents.__dict__["get_llm_agent_class"] = None
    sys.modules["core.llm_agents"] = llm_agents
    from core.judge import with_shared_rules

    trial_path = UPSTREAM / "core/trial.py"
    trial = ast.parse(trial_path.read_text(encoding="utf-8"))
    render = next(
        node
        for node in trial.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_prompt"
    )
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[render], type_ignores=[]), str(trial_path), "exec"), namespace)

    with tempfile.TemporaryDirectory() as directory, tarfile.open(args.archive) as archive:
        scratch = Path(directory)
        archived_rows = archive.extractfile("data.jsonl")
        assert archived_rows is not None
        rows = archived_rows.read().splitlines()
        data_path = scratch / "data.jsonl"
        data_path.write_bytes(b"\n".join(rows) + b"\n")
        judge_path = scratch / "judge_schema.py"
        judge_path.write_bytes((original / "judge_schema.py").read_bytes())
        judge = runpy.run_path(str(judge_path))

    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "extra_context.json").write_text(
        json.dumps(judge["EXTRA_CONTEXT"], ensure_ascii=False) + "\n", encoding="utf-8"
    )
    schema = with_shared_rules(judge["Verdict"]).model_json_schema()
    (DESTINATION / "shared_verdict_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    prompt = namespace["render_prompt"](original / "prompts/instruction.md", json.loads(rows[0]))
    (DESTINATION / "row0_prompt.txt").write_text(prompt, encoding="utf-8")


if __name__ == "__main__":
    main()
