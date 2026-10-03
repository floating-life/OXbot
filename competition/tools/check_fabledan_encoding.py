#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare C++ FableDan history/action encoding with the Python reference.

Run from the repository root after tools/build.ps1 or tools/build_wsl.sh::

    competition/.venv/Scripts/python.exe competition/tools/check_fabledan_encoding.py \
        --probe bin/core_probe --report reports/fabledan_encoding_parity.json

These are synthetic encoder inputs, not completed games or a protocol replay.
Candidate-set coverage and network numerical parity have separate checkers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "competition"))
sys.path.insert(0, str(ROOT / "tools"))

from fabledan.cards import level_str_botzone, rank_of
from fabledan.combos import claim_ids, gen_moves
from fabledan.encode import FEAT_DIM, encode_decision
from probe import Probe


def _event_json(event):
    kind, player = event[:2]
    if kind == "play":
        cards, claims = list(event[2].cards), claim_ids(event[2])
    elif kind == "pass":
        cards, claims = [], []
    else:
        raise ValueError("exchanges require physical-card history")
    return {"player": player, "stage": "play", "action": cards, "claim": claims}


def _case(name, player, level, hand, legal, events, lead=None, history=None,
          level_alias=None, counts=None, done=None, tribute=0, resist=False):
    counts = [27] * 4 if counts is None else counts
    done = [False] * 4 if done is None else done
    observation = {
        "player": player, "level": level, "hand": hand, "legal": legal,
        "lead": lead, "events": events, "done": done, "left": counts,
    }
    tokens, features = encode_decision(observation)
    cpp_observation = {
        "player": player, "level": level_alias or level_str_botzone(level),
        "hand": hand, "leading": lead is None, "tribute": tribute,
        "resist": resist, "remaining_counts": counts,
        "history": [_event_json(event) for event in events] if history is None else history,
        "done": [index for index, value in enumerate(done) if value],
    }
    request = {
        "command": "fabledan_features", "observation": cpp_observation,
        "moves": [[list(move.cards), claim_ids(move)] for move in legal],
    }
    return name, request, tokens, features


def _cases(seed, count):
    rng = random.Random(seed)
    for index in range(count):
        deck = list(range(108))
        rng.shuffle(deck)
        hand, level, player = deck[:27], rng.randrange(13), index % 4
        lead, events = None, []
        if index % 3:
            lead = next(move for move in gen_moves(deck[27:54], level, None) if move.type)
            if index % 3 == 2:
                events.append(("pass", (player + 1) % 4))
            events.append(("play", (player + index % 3) % 4, lead))
        legal = gen_moves(hand, level, lead)
        yield _case(f"random_{index:03d}", player, level, hand, legal, events, lead)

    # Retain the original eight exchange regression cases and their seed.
    exchange_rng = random.Random(11)
    for index in range(8):
        level = exchange_rng.randrange(13)
        hand = list(range(27))
        exchange_rng.shuffle(hand)
        events, history = [], []
        resist = index == 1
        for kind, player, card in [("tribute", 3, hand[0])] + (
            [("return", 0, hand[1])] if index % 2 else []
        ):
            cards = [] if resist else [card]
            history.append({"player": player, "stage": kind, "action": cards, "claim": cards})
            if not resist:
                events.append((kind, player, rank_of(card)))
        yield _case(f"exchange_{index}", index % 4, level, hand, [], events,
                    history=history, tribute=1, resist=resist)

    # Exercise the exact 512-token reference window plus ten-level aliases,
    # relative remaining-card counts, and finished-player feature flags.
    for viewer, alias in enumerate(("0", "10", "T", "t")):
        hand = [36, 37, 90, 52, 53, 106, 107]
        legal = gen_moves(hand, 9, None)
        events = [("pass", index % 4) for index in range(300)]
        counts = [0, 13, 22, 27]
        counts[viewer] = len(hand)
        done = [value == 0 for value in counts]
        yield _case(f"history_window_alias_{alias}", viewer, 9, hand, legal, events,
                    level_alias=alias, counts=counts, done=done)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--random-cases", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.random_cases < 1:
        parser.error("--random-cases must be positive")

    checks = []
    with Probe(args.probe.resolve(), timeout=args.timeout) as probe:
        for name, request, tokens, features in _cases(args.seed, args.random_cases):
            actual = probe.call(**request)
            rows = np.asarray(actual.get("actions", []), dtype=np.float32)
            if rows.size == 0:
                rows = rows.reshape(0, FEAT_DIM)
            shape_matches = rows.shape == features.shape
            exact_features = shape_matches and np.array_equal(rows, features)
            error = float(np.max(np.abs(rows - features))) if shape_matches and rows.size else 0.0
            check = {
                "name": name,
                "tokens": len(tokens),
                "candidates": len(features),
                "tokens_match": actual.get("tokens") == tokens,
                "feature_shape_matches": shape_matches,
                "feature_float32_exact": bool(exact_features),
                "max_absolute_error": error if shape_matches else None,
                "error": actual.get("error"),
            }
            check["passed"] = bool(check["tokens_match"] and exact_features and not check["error"])
            checks.append(check)

    passed = all(check["passed"] for check in checks)
    source_paths = (
        "competition/tools/check_fabledan_encoding.py", "competition/fabledan/encode.py",
        "competition/fabledan/cards.py", "competition/fabledan/combos.py",
        "core/src/fabledan_features.cpp", "core/include/oxbot/card.hpp", "tools/core_probe.cpp",
    )
    report = {
        "status": "passed" if passed else "failed",
        "scope": "synthetic history/action encoder parity; not game, protocol, or strength acceptance",
        "seed": args.seed,
        "random_cases": args.random_cases,
        "exchange_seed": 11,
        "case_count": len(checks),
        "feature_dim": FEAT_DIM,
        "total_candidate_rows": sum(check["candidates"] for check in checks),
        "probe_sha256": _sha(args.probe),
        "source_sha256": {path: _sha(ROOT / path) for path in source_paths},
        "cases": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
