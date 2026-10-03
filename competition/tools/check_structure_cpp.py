#!/usr/bin/env python3
"""Strict ordered candidate and float32 feature parity for structure-v2.

Run after compiling the current core_probe with feature_dim=224 support::

    python competition/tools/check_structure_cpp.py --probe bin/core_probe \
        --report reports/structure_cpp_parity.json --deals-per-level 2

This validates the bounded representative candidate contract, not exhaustive
physical-card subset coverage, inference Q-values, or playing strength. Legacy
80-dimensional acceptance remains in check_fabledan_candidates.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
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
from fabledan.combos import Move, PASS, TYPE_NAMES, beats, claim_ids, classify_claim, gen_moves
from fabledan.encode import STRUCTURE_FEAT_DIM, encode_decision
from probe import Probe


def semantic_key(move):
    return move.type, move.key, move.size, tuple(sorted(move.claim_ranks))


def candidate_key(move):
    return semantic_key(move) + (tuple(sorted(card % 54 for card in move.cards)),)


def fixture_hands():
    return [
        ("double_straight_flush", 1, [8, 12, 16, 20, 24, 9, 13, 17, 21, 25, 28]),
        ("distinct_suit_singles", 1, [8, 9, 12, 16, 20, 24]),
        ("duplicate_deck_faces", 1, [8, 62, 9, 63, 12, 66, 16, 70]),
        ("high_ace_wildcards_rocket", 1, [36, 40, 44, 48, 0, 4, 58, 52, 106, 53, 107]),
        ("budget_256", 1, [4, 58] + [rank * 4 + suit for rank in range(2, 13) for suit in (0, 1)] + [0, 1, 2]),
    ]


def _observation(hand, level, player=0, lead=None, lead_owner=None):
    left = [27] * 4
    left[player] = len(hand)
    if lead is not None:
        left[lead_owner] -= lead.size
    return {"hand": list(hand), "level": level, "player": player,
            "lead": lead, "lead_owner": lead_owner, "left": left,
            "done": [False] * 4,
            "events": [] if lead is None else [("play", lead_owner, lead)]}


def _request(observation):
    history = [{"player": event[1], "stage": "play", "action": list(event[2].cards),
                "claim": claim_ids(event[2])} for event in observation["events"]]
    return {"command": "fabledan_candidates", "feature_dim": STRUCTURE_FEAT_DIM,
            "observation": {"player": observation["player"],
                            "level": level_str_botzone(observation["level"]),
                            "hand": observation["hand"], "leading": observation["lead"] is None,
                            "remaining_counts": observation["left"], "history": history,
                            "done": [seat for seat, finished in enumerate(observation["done"]) if finished],
                            "tribute": 0, "resist": False}}


def build_cases(seed=20261007, deals_per_level=2):
    """All 13 levels; each base case also gets hand-order and deck-swap checks."""
    rng = random.Random(seed)
    bases = []
    for level in range(13):
        for deal in range(deals_per_level):
            deck = list(range(108))
            rng.shuffle(deck)
            hand, opponent = deck[:27], deck[27:54]
            player = (level + deal) % 4
            owner = (player + (2 if deal % 2 else 3)) % 4
            leads = gen_moves(opponent, level, None, feature_dim=224)
            lead = rng.choice(leads)
            for following in (False, True):
                name = f"random_level{level}_deal{deal}_{'follow' if following else 'lead'}"
                bases.append((name, _observation(hand, level, player, lead if following else None,
                                                 owner if following else None)))
    for name, level, hand in fixture_hands():
        bases.append((name, _observation(hand, level)))
        available = [card for card in range(108) if card not in hand and not is_wildcard(card, level)]
        lead = gen_moves([available[0]], level, None, feature_dim=224)[0]
        bases.append((name + "_follow", _observation(hand, level, lead=lead, lead_owner=1)))
    for name, original in bases:
        expected = None
        for variant in ("original", "reordered", "deck_swapped"):
            observation = copy.deepcopy(original)
            if variant == "reordered":
                rng.shuffle(observation["hand"])
            elif variant == "deck_swapped":
                observation["hand"] = [(card + 54) % 108 for card in observation["hand"]]
                if observation["lead"] is not None:
                    old = observation["lead"]
                    lead = Move(old.type, old.key, [(card + 54) % 108 for card in old.cards], list(old.claim_ranks))
                    observation["lead"] = lead
                    observation["events"] = [("play", observation["lead_owner"], lead)]
            observation["legal"] = gen_moves(observation["hand"], observation["level"],
                                               observation["lead"], feature_dim=224)
            keys = list(map(candidate_key, observation["legal"]))
            if expected is None:
                expected = keys
            _, features = encode_decision(observation, feat_dim=224)
            yield {"name": name, "variant": variant, "observation": observation,
                   "moves": observation["legal"], "features": features,
                   "python_invariance_passed": keys == expected, "request": _request(observation)}


def _physical_claim_error(cards, claims, hand, level):
    if len(cards) != len(claims) or len(cards) != len(set(cards)) or not set(cards).issubset(hand):
        return "action_size_or_physical_ownership_invalid"
    claimed = Counter(card % 54 for card in claims)
    naturals = Counter(card % 54 for card in cards if not is_wildcard(card, level))
    if any(claimed[face] < amount for face, amount in naturals.items()):
        return "claim_changes_a_nonwild_physical_face"
    if any(claimed[face] > naturals[face] for face in (52, 53)):
        return "claim_substitutes_a_joker"
    return None


def feature_name(column):
    for start, stop, label in ((0, 15, "hand_rank"), (80, 134, "hand_face"),
                               (134, 188, "remaining_face"), (188, 203, "remaining_natural_rank"),
                               (205, 213, "remaining_rank_count_threshold")):
        if start <= column < stop:
            return f"{label}[{column - start}]"
    return {203: "remaining_wildcards", 204: "remaining_hand_size", 213: "natural_straight_windows",
            214: "wild_straight_windows", 215: "natural_straight_flush_windows", 216: "wild_straight_flush_windows",
            217: "natural_triple_pair_windows", 218: "wild_triple_pair_windows", 219: "natural_two_triples_windows",
            220: "wild_two_triples_windows", 221: "full_house_rank_pairs", 222: "maximum_bomb_size",
            223: "four_joker_rocket"}.get(column, f"legacy_feature[{column}]")


def compare_case(case, actual, atol=0.0):
    observation = case["observation"]
    python_moves = case["moves"]
    check = {"name": case["name"], "variant": case["variant"],
             "level": level_str_botzone(observation["level"]),
             "mode": "leading" if observation["lead"] is None else "following",
             "python_candidates": len(python_moves), "cpp_candidates": len(actual.get("moves", [])),
             "python_invariance_passed": case["python_invariance_passed"]}
    try:
        if actual.get("error") or "moves" not in actual or "actions" not in actual:
            raise ValueError(f"probe returned no candidate/features: {actual}")
        cpp_moves = []
        for index, (cards, claims) in enumerate(actual["moves"]):
            error = _physical_claim_error(cards, claims, observation["hand"], observation["level"])
            if error:
                raise ValueError(f"C++ candidate {index}: {error}")
            move = classify_claim(cards, claims, observation["level"])
            if move.type == PASS and observation["lead"] is None:
                raise ValueError(f"C++ candidate {index}: pass while leading")
            if move.type != PASS and not beats(move, observation["lead"], observation["level"]):
                raise ValueError(f"C++ candidate {index}: cannot beat previous move")
            cpp_moves.append(move)
        py_keys, cpp_keys = list(map(candidate_key, python_moves)), list(map(candidate_key, cpp_moves))
        check["ordered_candidates_match"] = py_keys == cpp_keys
        check["cpp_candidate_keys_unique"] = len(cpp_keys) == len(set(cpp_keys))
        mismatches = [{"index": index, "python": py_keys[index] if index < len(py_keys) else None,
                       "cpp": cpp_keys[index] if index < len(cpp_keys) else None}
                      for index in range(max(len(py_keys), len(cpp_keys)))
                      if index >= len(py_keys) or index >= len(cpp_keys) or py_keys[index] != cpp_keys[index]]
        if mismatches:
            check["candidate_mismatch_samples"] = mismatches[:5]
        py_features = case["features"]
        cpp_features = np.asarray(actual["actions"], dtype=np.float32)
        if cpp_features.size == 0:
            cpp_features = cpp_features.reshape(0, 224)
        check["python_feature_shape"] = list(py_features.shape)
        check["cpp_feature_shape"] = list(cpp_features.shape)
        if cpp_features.shape != (len(cpp_moves), 224):
            raise ValueError("C++ did not return one 224-dimensional feature row per move")
        if not np.isfinite(cpp_features).all():
            raise ValueError("C++ returned nonfinite feature values")
        check["features_match"] = (cpp_features.shape == py_features.shape
                                     and bool(np.allclose(cpp_features, py_features, atol=atol, rtol=0)))
        common = min(len(cpp_features), len(py_features))
        if common:
            delta = np.abs(cpp_features[:common].astype(np.float64) - py_features[:common].astype(np.float64))
            row, column = map(int, np.unravel_index(np.argmax(delta), delta.shape))
            check["max_feature_abs_error"] = float(delta[row, column])
            check["worst_feature_misalignment"] = {
                "candidate_index": row, "feature_column": column, "feature_name": feature_name(column),
                "python": float(py_features[row, column]), "cpp": float(cpp_features[row, column]),
                "abs_error": float(delta[row, column]), "candidate_keys_match": py_keys[row] == cpp_keys[row],
                "python_key": py_keys[row], "cpp_key": cpp_keys[row],
            }
        check["passed"] = bool(check["ordered_candidates_match"] and check["features_match"]
                               and check["cpp_candidate_keys_unique"] and check["python_invariance_passed"])
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        check.update(passed=False, error=str(exc))
    if not check["passed"]:
        check["reproduction_request"] = case["request"]
    return check


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = {"competition/tools/check_structure_cpp.py", "competition/fabledan/encode.py",
             "competition/fabledan/combos.py", "competition/fabledan/cards.py",
             "tools/core_probe.cpp", "core/src/rules.cpp", "core/src/policy.cpp", "core/include/oxbot/card.hpp"}
    paths.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "core/src").glob("fabledan*.cpp"))
    paths.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "core/include/oxbot").glob("fabledan*.hpp"))
    return {path: _sha(ROOT / path) for path in sorted(paths)}


def run_check(probe_path, *, seed=20261007, deals_per_level=2, timeout=30, atol=0.0):
    if deals_per_level < 1 or not np.isfinite(atol) or atol < 0:
        raise ValueError("positive deals_per_level and nonnegative finite atol required")
    probe_path = Path(probe_path).resolve()
    before_probe, before_sources = _sha(probe_path), source_hashes()
    checks, candidate_types = [], Counter()
    with Probe(probe_path, timeout=timeout) as probe:
        for case in build_cases(seed, deals_per_level):
            actual = probe.call(**case["request"])
            checks.append(compare_case(case, actual, atol))
            candidate_types.update(TYPE_NAMES[move.type] for move in case["moves"])
    stable = before_probe == _sha(probe_path) and before_sources == source_hashes()
    worst = max((check for check in checks if "max_feature_abs_error" in check),
                key=lambda check: check["max_feature_abs_error"], default=None)
    passed = stable and all(check["passed"] for check in checks)
    return {"schema": "fabledan-structure-cpp-parity-v1", "status": "passed" if passed else "failed",
            "feature_dim": 224, "seed": seed, "deals_per_level": deals_per_level,
            "all_level_ranks": list(range(13)), "case_count": len(checks),
            "passed_cases": sum(check["passed"] for check in checks), "atol": atol, "rtol": 0,
            "ordered_key": ["type", "comparison_key", "size", "sorted_claim_ranks", "sorted_action_mod54_faces"],
            "scope": "strict candidate order and feature rows; no Q-value, exhaustive subset, or playing-strength claim",
            "python_candidate_types": dict(sorted(candidate_types.items())),
            "python_candidates": sum(check["python_candidates"] for check in checks),
            "cpp_candidates": sum(check["cpp_candidates"] for check in checks),
            "probe_sha256": before_probe, "source_sha256": before_sources,
            "python_encoder_sha256": before_sources["competition/fabledan/encode.py"],
            "python_candidate_generator_sha256": before_sources["competition/fabledan/combos.py"],
            "cpp_runtime_sha256": {path: digest for path, digest in before_sources.items() if path.startswith("core/")},
            "inputs_unchanged_during_check": stable,
            "worst_feature_misalignment": None if worst is None else {
                "case_name": worst["name"], "variant": worst["variant"], **worst["worst_feature_misalignment"]},
            "cases": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--deals-per-level", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    try:
        report = run_check(args.probe, seed=args.seed, deals_per_level=args.deals_per_level,
                           timeout=args.timeout, atol=args.atol)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        report = {"schema": "fabledan-structure-cpp-parity-v1", "status": "failed", "error": str(exc)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": report.get("case_count", 0),
                      "passed_cases": report.get("passed_cases", 0), "report": str(args.report)}))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
