#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check sampled Python/C++ FableDan candidate semantics and feature groups.

From the repository root, after building bin/core_probe::

    competition/.venv/Scripts/python.exe competition/tools/check_fabledan_candidates.py \
        --probe bin/core_probe --report reports/fabledan_candidate_parity.json

Each shuffled deal yields one leading and one following test. The structural
key is (type, size, sorted claim-rank multiset). For feature comparison, both
sides group 80-dimensional float32 rows with index 65 zeroed and keep the row
with the fewest wildcards, matching the C++ policy's documented grouping.
This does not assert equal physical cards, suit representatives, candidate
order, chosen moves, or all ungrouped Python feature variants.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "competition"))
sys.path.insert(0, str(ROOT / "tools"))

from fabledan.cards import is_wildcard, level_str_botzone
from fabledan.combos import TYPE_NAMES, claim_ids, classify_claim, gen_moves
from fabledan.encode import FEAT_DIM, encode_decision
from probe import Probe


def _semantic(move):
    return move.type, move.size, tuple(sorted(move.claim_ranks))


def _feature_groups(rows):
    groups = {}
    for row in rows:
        semantic = row.copy()
        semantic[65] = 0
        key = semantic.tobytes()
        if key not in groups or row[65] < groups[key][65]:
            groups[key] = row
    return {row.tobytes() for row in groups.values()}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _requests(seed, deals):
    rng = random.Random(seed)
    for index in range(deals):
        deck = list(range(108))
        rng.shuffle(deck)
        hand, opponent = deck[:27], deck[27:54]
        level, player = rng.randrange(13), index % 4
        opponent_moves = [move for move in gen_moves(opponent, level, None) if move.type]
        sampled_lead = rng.choice(opponent_moves)
        for following in (False, True):
            lead = sampled_lead if following else None
            legal = gen_moves(hand, level, lead)
            owner = (player + 3) % 4
            events = [("play", owner, lead)] if following else []
            observation = {
                "player": player, "level": level, "hand": hand, "legal": legal,
                "lead": lead, "lead_owner": owner if following else None,
                "events": events, "done": [False] * 4, "left": [27] * 4,
            }
            _, features = encode_decision(observation)
            history = [{"player": owner, "stage": "play", "action": list(lead.cards),
                        "claim": claim_ids(lead)}] if following else []
            request = {
                "command": "fabledan_candidates",
                "observation": {
                    "player": player, "level": level_str_botzone(level), "hand": hand,
                    "leading": not following, "tribute": 0, "resist": False,
                    "remaining_counts": [27] * 4, "history": history, "done": [],
                },
            }
            yield index, following, level, legal, features, request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--deals", type=int, default=100,
                        help="each deal yields one lead and one follow test")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.deals < 1:
        parser.error("--deals must be positive")

    source_paths = (
        "competition/tools/check_fabledan_candidates.py",
        "competition/fabledan/combos.py", "competition/fabledan/cards.py",
        "competition/fabledan/encode.py", "core/src/rules.cpp",
        "core/src/fabledan_features.cpp", "core/src/policy.cpp",
        "core/include/oxbot/card.hpp", "tools/core_probe.cpp",
    )
    # Snapshot the executable hash before it starts; another build may replace
    # the file on disk while an already-running WSL process still uses it.
    probe_sha = _sha(args.probe)
    source_hashes = {path: _sha(ROOT / path) for path in source_paths}
    checks, lead_types, candidate_types = [], Counter(), Counter()
    with Probe(args.probe.resolve(), timeout=args.timeout) as probe:
        for index, following, level, py_moves, py_features, request in _requests(args.seed, args.deals):
            actual = probe.call(**request)
            check = {
                "deal": index, "mode": "following" if following else "leading",
                "level": level_str_botzone(level),
                "hand_wildcards": sum(is_wildcard(card, level) for card in request["observation"]["hand"]),
                "python_raw_candidates": len(py_moves), "cpp_raw_candidates": len(actual.get("moves", [])),
            }
            try:
                if actual.get("error") or "moves" not in actual or "actions" not in actual:
                    raise ValueError(f"probe did not return candidates: {actual}")
                cpp_moves = [classify_claim(move[0], move[1], level) for move in actual["moves"]]
                cpp_features = np.asarray(actual["actions"], dtype=np.float32)
                if cpp_features.size == 0:
                    cpp_features = cpp_features.reshape(0, FEAT_DIM)
                if cpp_features.shape != (len(cpp_moves), FEAT_DIM):
                    raise ValueError(f"bad C++ feature shape: {cpp_features.shape}")
                py_set, cpp_set = set(map(_semantic, py_moves)), set(map(_semantic, cpp_moves))
                py_groups, cpp_groups = _feature_groups(py_features), _feature_groups(cpp_features)
                check.update({
                    "python_semantic_groups": len(py_set), "cpp_semantic_groups": len(cpp_set),
                    "missing_semantic_groups": len(py_set - cpp_set),
                    "extra_semantic_groups": len(cpp_set - py_set),
                    "python_feature_groups": len(py_groups), "cpp_feature_groups": len(cpp_groups),
                    "missing_feature_groups": len(py_groups - cpp_groups),
                    "extra_feature_groups": len(cpp_groups - py_groups),
                    "semantic_sets_match": py_set == cpp_set,
                    "canonical_feature_sets_match": py_groups == cpp_groups,
                    "passed": py_set == cpp_set and py_groups == cpp_groups,
                })
                if not check["passed"]:
                    check["missing_semantic_samples"] = sorted(py_set - cpp_set)[:5]
                    check["extra_semantic_samples"] = sorted(cpp_set - py_set)[:5]
                    check["missing_feature_samples"] = [np.frombuffer(row, dtype=np.float32).tolist()
                                                        for row in sorted(py_groups - cpp_groups)[:2]]
                    check["extra_feature_samples"] = [np.frombuffer(row, dtype=np.float32).tolist()
                                                      for row in sorted(cpp_groups - py_groups)[:2]]
                candidate_types.update(TYPE_NAMES[move.type] for move in py_moves)
                if following:
                    event = request["observation"]["history"][0]
                    lead_types[TYPE_NAMES[classify_claim(event["action"], event["claim"], level).type]] += 1
            except (ValueError, KeyError, TypeError) as exc:
                check.update({"passed": False, "error": str(exc)})
            if not check["passed"]:
                check["reproduction_request"] = request
            checks.append(check)

    passed = all(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed else "failed", "seed": args.seed, "deals": args.deals,
        "leading_cases": args.deals, "following_cases": args.deals, "case_count": len(checks),
        "passed_cases": sum(check["passed"] for check in checks),
        "scope": "sampled candidate semantics and canonical float32 feature sets; not match or strength acceptance",
        "semantic_key": ["type", "size", "sorted claim-rank multiset"],
        "feature_grouping": "zero 0-based column 65, group exact float32 rows, retain minimum wildcard count on both sides",
        "not_compared": ["physical card ids", "suit representatives", "candidate order",
                         "selected moves", "all ungrouped Python feature variants"],
        "python_raw_candidates": sum(check["python_raw_candidates"] for check in checks),
        "cpp_raw_candidates": sum(check["cpp_raw_candidates"] for check in checks),
        "following_lead_type_counts": dict(sorted(lead_types.items())),
        "python_candidate_type_counts": dict(sorted(candidate_types.items())),
        "probe_sha256": probe_sha, "source_sha256": source_hashes, "cases": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
