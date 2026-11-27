# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/tasks/chess/game/board.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

"""Positions and moves: a validated FEN, SAN tokens, one move parsed against a board."""

# Runs inside the Chess Docker image's isolated /opt/.svc/venv.
# pyright: reportMissingImports=false

from __future__ import annotations

import re

import chess


def validated_board(fen: str) -> chess.Board:
    if not isinstance(fen, str) or len(fen) > 200:
        raise ValueError("FEN must be a string of at most 200 characters")
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"Invalid chess position (status {int(board.status())})")
    return board


def san_tokens(sequence: str) -> list[str]:
    if not isinstance(sequence, str) or len(sequence) > 16384:
        raise ValueError("SAN sequence must be a string of at most 16384 characters")
    result: list[str] = []
    ended = False
    for raw_token in sequence.split():
        token = re.sub(r"^\d+\.(?:\.\.)?", "", raw_token)
        if not token:
            continue
        if ended:
            raise ValueError("Moves after a PGN result marker are not accepted")
        if token in {"1-0", "0-1", "1/2-1/2", "*"}:
            ended = True
            continue
        result.append(token)
    if len(result) > 256:
        raise ValueError("At most 256 plies are accepted")
    return result


def parse_move(board: chess.Board, token: str) -> chess.Move:
    if not isinstance(token, str) or len(token) > 20 or len(token.split()) != 1:
        raise ValueError("Submit one SAN move, for example Nf3 or Qxh7+")
    token = token.replace("0", "O")
    move = board.parse_san(token)
    if not move or move not in board.legal_moves:
        raise ValueError("Null and illegal moves are not accepted")
    # Allow omitted check/mate suffixes, but reject coordinate/UCI and long algebraic.
    if board.san(move).rstrip("+#") != token.rstrip("+#"):
        raise ValueError("Move must use SAN, not coordinate notation")
    return move
