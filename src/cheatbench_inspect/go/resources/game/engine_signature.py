# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/tasks/go/engine_signature.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

"""Verify KataGo answers the two recorded signature probes within tolerance."""

# Runs inside the Go engine container's /opt/engine/venv.
# pyright: reportMissingImports=false

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from game.go_game import GoGame, Settings

SIGNATURE_PATH = Path(__file__).resolve().parents[1] / "engine_signature.json"
PROBES_DIR = Path(__file__).resolve().parents[1] / "probes"
PROBES = ("ogs_68506069", "ogs_63091411")
VISITS = 400
EXACT = 1e-9
COMPATIBLE = 1e-3


def engine_version(binary: str) -> str:
    try:
        out = subprocess.run(
            [binary, "version"], capture_output=True, text=True, timeout=30, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    version = next((line for line in lines if line.startswith("KataGo")), "unknown")
    backend = next(
        (
            line.split("backend")[0].replace("Using", "").strip()
            for line in lines
            if "backend" in line
        ),
        "unknown",
    )
    return f"{version} ({backend})"


async def probe(engine: Any, *, probes_dir: Path | None = None) -> dict[str, Any]:
    """The engine's version and, for each probe position, its root numbers and top move."""
    answers = []
    fixtures = probes_dir or PROBES_DIR
    for fid in PROBES:
        fixture = json.loads((fixtures / f"tournament_{fid}.json").read_text(encoding="utf-8"))
        game = GoGame(fixture, engine, Settings())
        analysis = await engine.analyse(
            game.move_list,
            initial_stones=game.stones,
            visits=VISITS,
            board_size=game.size,
            komi=game.komi,
            rules=game.rules,
        )
        root = analysis.get("rootInfo") or {}
        top = sorted(analysis.get("moveInfos", []), key=lambda m: m.get("order", 99))
        answers.append(
            {
                "position": fid,
                "visits": VISITS,
                "winrate": root.get("winrate"),
                "score_lead": root.get("scoreLead"),
                "top_move": top[0].get("move") if top else None,
            }
        )
    return {
        "engine": engine_version(engine.katago),
        "network": Path(engine.network).name,
        "settings": engine.settings,
        "probes": answers,
    }


async def verify(
    engine: Any,
    *,
    signature_path: Path | None = None,
    probes_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare the live engine with the recorded signature."""
    sig_path = signature_path or SIGNATURE_PATH
    if not sig_path.is_file():
        raise RuntimeError(f"{sig_path} is missing; cannot verify Go engine signature.")
    expected = json.loads(sig_path.read_text(encoding="utf-8"))
    live = await probe(engine, probes_dir=probes_dir)
    deltas = [
        abs(float(e["winrate"]) - float(lv["winrate"]))
        for e, lv in zip(expected["probes"], live["probes"], strict=True)
    ]
    moves_agree = all(
        e["top_move"] == lv["top_move"]
        for e, lv in zip(expected["probes"], live["probes"], strict=True)
    )
    worst = max(deltas)
    if live["network"] != expected["network"]:
        raise RuntimeError(
            f"The engine loaded {live['network']}; the positions were certified on {expected['network']}."
        )
    if worst > COMPATIBLE or not moves_agree:
        raise RuntimeError(
            f"The engine ({live['engine']}) answers the signature probes {worst:.2e} in win rate "
            f"away from the recorded {expected['engine']}, or prefers a different move; the "
            "positions were certified on that engine and the results would not be comparable."
        )
    return {
        "status": "exact" if worst <= EXACT else "compatible",
        "max_winrate_delta": worst,
        "live_engine": live["engine"],
        "recorded_engine": expected["engine"],
        "probes": live["probes"],
    }
