#!/opt/engine/venv/bin/python
"""Deterministic CPU-only KataGo analysis protocol test double for local Docker tests.

Implements the exact `katago version` and `katago analysis` JSON-line stdin/stdout
protocol, including `allowMoves` forced searches, the two recorded signature probes
(`ogs_68506069` and `ogs_63091411`), and the certified winning reference line for
`ogs_69046172` (`B:F8`, `W:J10`, `B:K9` -> resignation).
"""

from __future__ import annotations

import json
import sys
from typing import Any

COLS = "ABCDEFGHJKLMNOPQRST"


def _side_to_move(moves: list[list[str]]) -> str:
    if not moves:
        return "B"
    return "W" if moves[-1][0].upper() == "B" else "B"


def _occupied(moves: list[list[str]], initial_stones: list[list[str]]) -> set[str]:
    pts: set[str] = set()
    for _, pt in initial_stones:
        if pt.lower() != "pass":
            pts.add(pt.upper())
    for _, pt in moves:
        if pt.lower() != "pass":
            pts.add(pt.upper())
    return pts


def _free_points(occupied: set[str], size: int, preferred: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for pt in preferred:
        if pt not in occupied and pt not in out:
            out.append(pt)
    for col in COLS[:size]:
        for row in range(1, size + 1):
            pt = f"{col}{row}"
            if pt not in occupied and pt not in out:
                out.append(pt)
                if len(out) >= 6:
                    return out
    return out


def _info(move: str, winrate: float, score_lead: float, order: int, visits: int) -> dict[str, Any]:
    return {
        "move": move,
        "winrate": winrate,
        "scoreLead": score_lead,
        "order": order,
        "visits": visits,
        "pv": [move],
    }


def _evaluate(query: dict[str, Any]) -> dict[str, Any]:
    qid = query.get("id", "")
    moves = query.get("moves") or []
    initial = query.get("initialStones") or []
    size = int(query.get("boardXSize", 19))
    visits = int(query.get("maxVisits", 400))
    allow = query.get("allowMoves") or []
    forced_move = allow[0]["moves"][0] if allow and allow[0].get("moves") else None
    player = _side_to_move(moves)
    occ = _occupied(moves, initial)

    # Probe 1: ogs_68506069 (40 setup moves, first B:Q17, last W:D10)
    if len(moves) == 40 and moves[0] == ["B", "Q17"] and moves[-1] == ["W", "D10"]:
        w, s, m = 0.82801731, 4.10576768, "C11"
        if forced_move:
            return {
                "id": qid,
                "rootInfo": {"currentPlayer": player, "winrate": w, "scoreLead": s},
                "moveInfos": [
                    _info(
                        forced_move,
                        w if forced_move == m else 0.35,
                        s if forced_move == m else -2.0,
                        0,
                        visits,
                    )
                ],
            }
        return {
            "id": qid,
            "rootInfo": {"currentPlayer": player, "winrate": w, "scoreLead": s},
            "moveInfos": [_info(m, w, s, 0, visits)],
        }

    # Probe 2: ogs_63091411 (130 setup moves, first B:Q3, last W:L3)
    if len(moves) == 130 and moves[0] == ["B", "Q3"] and moves[-1] == ["W", "L3"]:
        w, s, m = 0.966636317, 4.47567632, "L2"
        if forced_move:
            return {
                "id": qid,
                "rootInfo": {"currentPlayer": player, "winrate": w, "scoreLead": s},
                "moveInfos": [
                    _info(
                        forced_move,
                        w if forced_move == m else 0.35,
                        s if forced_move == m else -2.0,
                        0,
                        visits,
                    )
                ],
            }
        return {
            "id": qid,
            "rootInfo": {"currentPlayer": player, "winrate": w, "scoreLead": s},
            "moveInfos": [_info(m, w, s, 0, visits)],
        }

    # Reference fixture: ogs_69046172 (76 setup moves, B:D4 ... W:G9)
    is_69046172 = len(moves) >= 76 and moves[0] == ["B", "D4"] and moves[75] == ["W", "G9"]
    if is_69046172:
        tail = [(c.upper(), p.upper()) for c, p in moves[76:]]
        if tail == []:
            # Start position: unique winning move F8.
            candidates = [("F8", 0.8713, 5.37), ("J10", 0.0081, -33.5), ("F12", 0.05, -20.0)]
            root_w, root_s = 0.8713, 5.37
        elif tail == [("B", "F8")]:
            # Opponent to reply: J10 keeps game alive (leaves only 1 winner K9).
            candidates = [("J10", 0.1334, -5.2), ("K9", 0.08, -9.0)]
            root_w, root_s = 0.1334, -5.2
        elif tail == [("B", "F8"), ("W", "J10")]:
            # After W:J10, Black has ONE unique winning move K9 (screen finds 1, confirm finds 1 -> no resign yet).
            candidates = [("K9", 0.8666, 5.1), ("B5", 0.4992, -0.4)]
            root_w, root_s = 0.8666, 5.1
        elif tail == [("B", "F8"), ("W", "J10"), ("B", "K9")]:
            # Opponent to reply after B:K9: every reply leaves Black with 2+ clear winners -> resignation!
            fallback = _free_points(occ, size, ("K10", "L10"))
            candidates = [(fallback[0], 0.05, -12.0)]
            root_w, root_s = 0.05, -12.0
        elif len(tail) == 4 and tail[:3] == [("B", "F8"), ("W", "J10"), ("B", "K9")]:
            # Trial opponent reply after B:K9: TWO clearly winning Black moves -> confirms resignation!
            pts = _free_points(occ, size, ("L10", "M10", "N10"))
            candidates = [(pts[0], 0.93, 9.5), (pts[1], 0.91, 8.2), (pts[2], 0.45, -1.0)]
            root_w, root_s = 0.93, 9.5
        else:
            # Any other branch (e.g. blunder): keep playing cleanly without resigning.
            pts = _free_points(occ, size, ("Q10", "R10", "P10"))
            candidates = [(pts[0], 0.55, 0.4), (pts[1], 0.52, 0.2)]
            root_w, root_s = 0.55, 0.4

        if forced_move:
            table = {m: (w, s) for m, w, s in candidates}
            w, s = table.get(forced_move.upper(), (0.30, -3.0))
            return {
                "id": qid,
                "rootInfo": {"currentPlayer": player, "winrate": root_w, "scoreLead": root_s},
                "moveInfos": [_info(forced_move, w, s, 0, visits)],
            }
        return {
            "id": qid,
            "rootInfo": {"currentPlayer": player, "winrate": root_w, "scoreLead": root_s},
            "moveInfos": [
                _info(m, w, s, idx, visits if idx == 0 else 10)
                for idx, (m, w, s) in enumerate(candidates)
            ],
        }

    # Generic fallback for warmup or edited boards.
    pts = _free_points(occ, size, ("Q16", "D4", "D16"))
    if forced_move:
        return {
            "id": qid,
            "rootInfo": {"currentPlayer": player, "winrate": 0.52, "scoreLead": 0.2},
            "moveInfos": [_info(forced_move, 0.52, 0.2, 0, visits)],
        }
    return {
        "id": qid,
        "rootInfo": {"currentPlayer": player, "winrate": 0.52, "scoreLead": 0.2},
        "moveInfos": [_info(m, 0.52 - 0.02 * i, 0.2, i, visits) for i, m in enumerate(pts[:2])],
    }


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "version":
        print("KataGo v1.18.2", flush=True)
        print("Using TestDouble backend", flush=True)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "analysis":
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                query = json.loads(line)
            except ValueError:
                continue
            reply = _evaluate(query)
            print(json.dumps(reply), flush=True)
        return
    raise SystemExit(f"Unsupported arguments: {sys.argv[1:]}")


if __name__ == "__main__":
    main()
