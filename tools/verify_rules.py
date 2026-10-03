"""Differential rule checks against the user's unmodified referee source."""
from __future__ import annotations
import argparse
import collections
import itertools
import json
from pathlib import Path
import random
import time

from oracle import Oracle, DEFAULT_JUDGE
from probe import Probe


def classify(rule, claim):
    try:
        return rule["checkPokerType"](claim)
    except (ValueError, IndexError):
        # Joker sequence/slice-overrun crashes are invalid generated actions;
        # the independent core must reject them instead of crashing.
        return "invalid", ()


def semantic(rule, move):
    kind, points = classify(rule, move[1])
    return tuple(sorted(c % 54 for c in move[0])), kind, str(points)


def brute_force(rule, hand, level):
    wild = int("A234567890JQK".index(level)) * 4
    result = set()
    for size in range(1, len(hand) + 1):
        for action in itertools.combinations(hand, size):
            positions = [i for i, card in enumerate(action) if card % 54 == wild]
            for replacements in itertools.product(range(52), repeat=len(positions)):
                claim = list(action)
                for index, card in zip(positions, replacements):
                    claim[index] = card
                if classify(rule, claim)[0] != "invalid":
                    result.add(semantic(rule, [list(action), claim]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=Path("bin/core_probe"))
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--claims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--report", type=Path, default=Path("reports/rules.json"))
    args = parser.parse_args()
    oracle = Oracle(args.judge)
    rng = random.Random(args.seed)
    stats = collections.Counter()
    started = time.monotonic()
    with Probe(args.probe) as probe:
        for level in "A234567890JQK":
            rule = oracle.rules(level)
            for _ in range(max(1, args.claims // 13)):
                n = rng.randint(0, 10)
                # Include syntactically possible repeated claim IDs: a claim is
                # a declared face multiset, not a physical allocation.
                claim = [rng.randrange(108) for _ in range(n)]
                expected = classify(rule, claim)[0]
                actual = probe.call(command="classify", claim=claim)["kind"]
                assert actual == expected, (level, claim, expected, actual)
                stats["random_claims"] += 1
            for _ in range(max(1, args.hands // 13)):
                hand = rng.sample(range(108), rng.randint(1, 27))
                moves = probe.call(command="generate", hand=hand, level=level, leading=True)["moves"]
                assert moves, ("no lead", hand)
                for action, claim in moves:
                    assert action and len(set(action)) == len(action) and set(action).issubset(hand)
                    assert rule["isLegalClaim"](action, claim, level, 0)
                    assert classify(rule, claim)[0] not in ("invalid", "pass"), (hand, level, action, claim)
                    assert probe.call(command="validate", hand=hand, level=level, leading=True,
                                      move=[action, claim])["ok"]
                    stats["generated_actions"] += 1
                for _ in range(min(50, len(moves))):
                    prev, curr = rng.choices(moves, k=2)
                    pa, ka = classify(rule, prev[1])
                    pb, kb = classify(rule, curr[1])
                    expected = rule["checkBigger"](pa, ka, pb, kb) is True
                    actual = probe.call(command="beats", previous=prev, move=curr, level=level)["ok"]
                    assert actual == expected, (level, prev, curr, expected, actual)
                    stats["comparisons"] += 1
                for command, fn in (("tribute", "isValidTribute"), ("return", "isValidReturn")):
                    accepted = probe.call(command=command, hand=hand, level=level)["cards"]
                    for c in accepted:
                        assert rule[fn](hand, c, level, 0), (level, command, hand, c)
                    stats[command + "_checks"] += len(accepted)
        # Small hands with no / one / two wildcards exercise generator recall.
        small_hands = [([0, 1, 2, 4, 5, 9], "7"), ([4, 8, 12, 16, 20], "2"),
                       ([4, 58, 8, 9, 10], "2"), ([0, 54, 52, 106, 4], "A")]
        for hand, level in small_hands:
            rule = oracle.rules(level)
            brute = brute_force(rule, hand, level)
            actual = {semantic(rule, m) for m in probe.call(command="generate", hand=hand,
                                                           level=level, leading=True)["moves"]}
            assert actual == brute, {"hand": hand, "level": level,
                                     "missing": sorted(brute - actual)[:10], "extra": sorted(actual - brute)[:10]}
            stats["exhaustive_hands"] += 1
            stats["exhaustive_semantic_actions"] += len(brute)
    report = {"oracle": str(oracle.path), "oracle_sha256": oracle.sha256,
              "seed": args.seed, "counts": dict(stats), "seconds": time.monotonic() - started,
              "status": "passed", "online_compatibility": "unverified"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
