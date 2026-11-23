"""Generate GeoGuessr judge parity fixtures from the original checkout; run from a scratch cwd."""

import importlib.util
import json
import sys
import types
from pathlib import Path

ORIGINAL = Path("/home/agent/cheatbench/tasks/geoguessr")
DEST = Path(__file__).resolve().parents[1] / "tests/cheatbench_inspect/geoguessr/fixtures"


def load_original_judge_schema() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "orig_geoguessr_judge", ORIGINAL / "judge_schema.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def composed_schema(judge: types.ModuleType) -> dict:
    sys.path.insert(0, "/home/agent/cheatbench")
    stub = types.ModuleType("core.llm_agents")
    stub.__dict__["get_llm_agent_class"] = None
    sys.modules["core.llm_agents"] = stub
    from core.judge import with_shared_rules

    return with_shared_rules(judge.Verdict).model_json_schema()


def main() -> None:
    judge = load_original_judge_schema()
    schema = composed_schema(judge)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "shared_verdict_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (DEST / "extra_context.txt").write_text(judge.EXTRA_CONTEXT, encoding="utf-8")


if __name__ == "__main__":
    main()
