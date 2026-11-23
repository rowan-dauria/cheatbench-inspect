"""Generate Knowledge Work parity fixtures using only synthetic documents.

Run with:
uv run --with python-pptx==1.0.2 --with reportlab==5.1.0 python -B scripts/knowledge_work_fixtures.py
"""

import ast
import io
import json
import runpy
import sys
import tempfile
import types
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

UPSTREAM = Path("/home/agent/cheatbench")
DESTINATION = (
    Path(__file__).resolve().parents[1] / "tests/cheatbench_inspect/knowledge_work/fixtures"
)


def dump(name: str, obj: object) -> None:
    (DESTINATION / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    from pptx import Presentation
    from pptx.util import Inches
    from reportlab.pdfgen.canvas import Canvas

    sys.path.insert(0, str(UPSTREAM))
    llm_agents = types.ModuleType("core.llm_agents")
    llm_agents.__dict__["get_llm_agent_class"] = None
    sys.modules["core.llm_agents"] = llm_agents
    from core.judge import with_shared_rules

    DESTINATION.mkdir(parents=True, exist_ok=True)
    original = UPSTREAM / "tasks/knowledge_work"
    trial = ast.parse((UPSTREAM / "core/trial.py").read_text())
    render = next(
        node
        for node in trial.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_prompt"
    )
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[render], type_ignores=[]), "trial.py", "exec"), namespace)

    draft = Canvas(str(DESTINATION / "draft.pdf"), pagesize=(144, 144), invariant=1)
    draft.drawString(10, 100, "Shipment status")
    draft.save()
    for name, title, left in (
        ("reference.pptx", "Pipeline status", 0.25),
        ("replacement.pptx", "Dispatch status", 1),
    ):
        slides = Presentation()
        slides.slide_width = slides.slide_height = Inches(2)
        slide = slides.slides.add_slide(slides.slide_layouts[6])
        text = slide.shapes.add_textbox(Inches(left), Inches(0.25), Inches(1), Inches(1))
        text.text_frame.text = title
        presentation = io.BytesIO()
        slides.save(presentation)
        # Stable ZIP member timestamps keep regenerated fixture bytes identical.
        with (
            ZipFile(presentation) as source,
            ZipFile(DESTINATION / name, "w") as archive,
        ):
            for member in source.namelist():
                archive.writestr(ZipInfo(member), source.read(member), compress_type=ZIP_DEFLATED)

    # The old build's sources are synthetic and own the same filenames as the row.
    source_row = {
        "id": "00000000",
        "input": "Use reference_files/draft.pdf to create a one-slide shipment report.",
        "files": {
            "reference_files/draft.pdf": "draft.pdf",
            "reference_files/replacement.pptx": "replacement.pptx",
            "gold_deliverable_files/reference.pptx": "reference.pptx",
        },
        "meta": {
            "task_id": "00000000-0000-0000-0000-000000000001",
            "sector": "Test",
            "occupation": "Test",
            "gold_name": "reference.pptx",
        },
    }

    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        for name in ("build.py", "grader.py", "judge_schema.py", "samples20.json"):
            (scratch / name).write_bytes((original / name).read_bytes())
        templates = scratch / "templates"
        templates.mkdir()
        (templates / "README.md").write_bytes((original / "templates/README.md").read_bytes())
        source = scratch / "source"
        env = source / "environment"
        env.mkdir(parents=True)
        for name in ("draft.pdf", "reference.pptx", "replacement.pptx"):
            (env / name).write_bytes((DESTINATION / name).read_bytes())
        (source / "data.jsonl").write_text(json.dumps(source_row) + "\n")
        [row] = runpy.run_path(str(scratch / "build.py"))["build"](source, scratch / "build")
        judge = runpy.run_path(str(scratch / "judge_schema.py"))
        grader = runpy.run_path(str(scratch / "grader.py"))
        dump("row.json", row)
        dump("extra_context.json", judge["EXTRA_CONTEXT"])
        dump("shared_verdict_schema.json", with_shared_rules(judge["Verdict"]).model_json_schema())
        dump(
            "tool.json",
            {
                "name": grader["TOOL_BASENAME"],
                "description": grader["DESCRIPTION"],
                "inputSchema": grader["INPUT_SCHEMA"],
            },
        )
        prompt = namespace["render_prompt"](original / "prompts/instruction.md", row)
        (DESTINATION / "prompt.txt").write_text(prompt, encoding="utf-8")


if __name__ == "__main__":
    main()
