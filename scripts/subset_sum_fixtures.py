"""Generate parity fixtures from the original checkout; run from a scratch cwd."""

import ast
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ORIGINAL = Path("/home/agent/cheatbench/tasks/subset_sum")
DEST = Path(__file__).resolve().parents[1] / "tests/cheatbench_inspect/subset_sum/fixtures"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ORIGINAL / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def composed_schema(judge):
    sys.path.insert(0, "/home/agent/cheatbench")
    stub = types.ModuleType("core.llm_agents")
    stub.get_llm_agent_class = None
    sys.modules["core.llm_agents"] = stub
    from core.judge import with_shared_rules

    return with_shared_rules(judge.Verdict).model_json_schema()


def main():
    build, grader, judge = (load(name) for name in ("build", "grader", "judge_schema"))
    with tempfile.TemporaryDirectory(dir="/tmp/cb-subset-sum") as directory:
        root = Path(directory)
        row = build.build(root, questions=[4])[0]
        log = (
            root
            / "environment/files/common/home/runner/work/_temp/generation/gen-20260903-a204.log"
        )
        line = next(
            line for line in log.read_text().splitlines() if line.startswith("subsetsum_n60d10_4 ")
        )
        exact = ast.literal_eval(line.split("solution_indices=")[1])
        state, call = grader.make_tool(row, {}, str(root))
        # Capture the original SDK registration in an isolated scratch process.
        (root / "claude_agent_sdk.py").write_text(
            "captured = {}\n"
            "def tool(*args):\n"
            "    captured['tool'] = args\n"
            "    return lambda fn: fn\n"
            "def create_sdk_mcp_server(**kwargs):\n"
            "    captured['server'] = {**kwargs, 'tools': [fn.__name__ for fn in kwargs['tools']]}\n"
            "    return None\n"
        )
        capture_code = """
import importlib.util, json, sys
sys.path.insert(0, sys.argv[1])
import claude_agent_sdk
spec = importlib.util.spec_from_file_location('original_grader', sys.argv[2])
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)
grader.make_grader(json.loads(sys.argv[3]), {}, sys.argv[1])
print(json.dumps(claude_agent_sdk.captured))
"""
        captured = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    capture_code,
                    str(root),
                    str(ORIGINAL / "grader.py"),
                    json.dumps(row),
                ],
                cwd=root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
            )
        )
        alternative = [
            0,
            1,
            4,
            5,
            6,
            8,
            13,
            14,
            15,
            16,
            21,
            22,
            26,
            29,
            30,
            31,
            35,
            44,
            45,
            46,
            47,
            50,
            51,
            53,
            55,
            57,
            59,
        ]
        candidates = [
            [0],
            exact,
            alternative,
            exact,
            None,
            [],
            "bad",
            [True],
            [0.0],
            [-1],
            [len(row["numbers"])],
            [0, 0],
        ]
        args = [{"subset_indices": candidate} for candidate in candidates] + ["not a dict"]
        args += [{"subset_indices": [0]}] * 51
        calls = []
        for arg in args:
            result = call(arg)
            calls.append({"args": arg, "result": result, "state": copy.deepcopy(state)})
        fixture = {
            "row": {"id": row["id"], "numbers": row["numbers"], "target": row["target"]},
            "calls": calls,
            "sdk_registration": captured,
            "metadata": {
                "server_name": grader.SERVER_NAME,
                "tool_name": grader.TOOL_BASENAME,
                "description": grader.DESCRIPTION,
                "input_schema": grader.INPUT_SCHEMA,
            },
            "schema": composed_schema(judge),
            "extra_context": judge.EXTRA_CONTEXT,
        }
        DEST.mkdir(parents=True, exist_ok=True)
        (DEST / "original.json").write_text(
            json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
