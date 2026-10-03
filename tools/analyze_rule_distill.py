"""Audit RulePolicy pseudo-label quality on the NJUPT development splits.

This is a read-only research diagnostic.  It opens exactly ``train.jsonl``
and ``validation.jsonl`` under the supplied ETL root, asks the audited C++
rule probe for legal candidates, and reproduces ``RulePolicy::choose`` in
Python.  It never opens ``test.jsonl`` and does not write training data or
change an online policy.  The resulting rule-vs-demonstration agreement is
the minimum evidence needed before considering rule distillation or an
uncertainty-triggered rule fallback.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from probe import Probe  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def face_multiset(cards: list[int]) -> tuple[int, ...]:
    return tuple(sorted(int(card) % 54 for card in cards))


def rank_index(card: int) -> int:
    face = int(card) % 54
    return face // 4 if face < 52 else -1


def is_level_card(card: int, level: str) -> bool:
    face = int(card) % 54
    return face < 52 and face % 4 == 0 and face // 4 == (9 if level == "10" else "A234567890JQK".index(level))


def is_joker(card: int) -> bool:
    return int(card) % 54 >= 52


def rule_choice(hand: list[int], level: str, moves: list[list[list[int]]], types: list[dict[str, Any]]) -> int:
    """Reproduce core/src/policy.cpp::RulePolicy::choose for play rows."""
    if not moves or len(moves) != len(types):
        raise ValueError("rule candidate metadata shape mismatch")
    hand_ranks = collections.Counter(rank_index(card) for card in hand if rank_index(card) >= 0)
    best = None
    best_cost = float("inf")
    for index, (move, meta) in enumerate(zip(moves, types, strict=True)):
        action = move[0]
        if not action:
            continue
        if len(action) == len(hand):
            return index
        kind = str(meta.get("kind", "invalid"))
        bomb = kind in {"bomb", "rocket", "straight_flush"}
        cost = -3.0 * len(action)
        if bomb:
            cost += 16.0
        for card in action:
            if is_level_card(card, level):
                cost += 2.5
            if is_joker(card):
                cost += 3.0
            rank = rank_index(card)
            if rank >= 0:
                cost += 0.04 * (13 if rank == 0 else rank)
                if not bomb and hand_ranks[rank] >= 4:
                    cost += 2.0
        if cost < best_cost:
            best = index
            best_cost = cost
    if best is not None:
        return best
    # On a legal following row RulePolicy returns the sole pass.  A leading
    # row has no pass; the generator should always contain a concrete move.
    return 0


def count_bin(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 8:
        return "4-8"
    if count <= 16:
        return "9-16"
    if count <= 32:
        return "17-32"
    if count <= 64:
        return "33-64"
    if count <= 128:
        return "65-128"
    if count <= 256:
        return "129-256"
    if count <= 512:
        return "257-512"
    return "513+"


def new_bucket() -> collections.Counter:
    return collections.Counter()


def audit_split(path: Path, probe: Probe, limit: int = 0) -> dict[str, Any]:
    totals = collections.Counter()
    by_context: dict[str, collections.Counter] = collections.defaultdict(new_bucket)
    by_kind: dict[str, collections.Counter] = collections.defaultdict(new_bucket)
    by_count: dict[str, collections.Counter] = collections.defaultdict(new_bucket)
    examples: list[dict[str, Any]] = []
    seen = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("stage") != "play":
                continue
            if limit and seen >= limit:
                break
            seen += 1
            features = row["features"]
            # The prepared BC corpus skips a follow row when the public
            # wildcard claim is unavailable.  RulePolicy cannot reconstruct
            # that comparison either, so count and skip it rather than
            # fabricating a previous claim.
            if not features["leading"] and features.get("last_claim") is None:
                totals["skipped_uncertain_previous"] += 1
                seen -= 1
                continue
            previous = None if features["leading"] else [features["last_cards"], features["last_claim"]]
            generated = probe.call(command="generate", hand=features["own_hand"],
                                   level=features["level_label"], leading=features["leading"],
                                   previous=previous, metadata=True)
            if generated.get("error"):
                raise RuntimeError(f"C++ rule generation failed for {row.get('game_id')}: {generated['error']}")
            moves, types = generated.get("moves"), generated.get("types")
            index = rule_choice(features["own_hand"], features["level_label"], moves, types)
            demonstrated = face_multiset(row["label"]["cards"])
            selected = face_multiset(moves[index][0])
            correct = selected == demonstrated
            context = "leading" if features["leading"] else "following"
            demo_size = len(row["label"]["cards"])
            kind = str(row["label"].get("kind", "unknown"))
            count_key = count_bin(len(moves))
            totals["rows"] += 1
            totals["correct"] += int(correct)
            totals["leading_rows" if features["leading"] else "following_rows"] += 1
            totals["leading_correct" if features["leading"] else "following_correct"] += int(correct)
            totals["demo_pass_rows"] += int(demo_size == 0)
            totals["rule_pass_rows"] += int(not moves[index][0])
            totals["rule_pass_correct"] += int(not moves[index][0] and demo_size == 0)
            totals[f"demo_size_{demo_size}"] += 1
            totals[f"rule_size_{len(moves[index][0])}"] += 1
            by_context[context]["rows"] += 1
            by_context[context]["correct"] += int(correct)
            by_kind[kind]["rows"] += 1
            by_kind[kind]["correct"] += int(correct)
            by_count[count_key]["rows"] += 1
            by_count[count_key]["correct"] += int(correct)
            if not correct and len(examples) < 16:
                examples.append({"game_id": row.get("game_id"), "deal_id": row.get("deal_id"),
                                 "event_index": row.get("event_index"), "leading": features["leading"],
                                 "candidate_count": len(moves), "demo": row["label"]["cards"],
                                 "rule": moves[index][0], "rule_kind": types[index].get("kind")})

    def finish(values: dict[str, collections.Counter]) -> dict[str, dict[str, Any]]:
        result = {}
        for key, value in sorted(values.items()):
            rows = int(value["rows"])
            result[key] = {"rows": rows, "correct": int(value["correct"]),
                           "agreement": int(value["correct"]) / max(1, rows)}
        return result

    rows = int(totals["rows"])
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": rows,
            "limit": limit, "totals": {**dict(totals), "agreement": totals["correct"] / max(1, rows),
                                         "leading_agreement": totals["leading_correct"] / max(1, totals["leading_rows"]),
                                         "following_agreement": totals["following_correct"] / max(1, totals["following_rows"])},
            "by_context": finish(by_context), "by_demonstrated_kind": finish(by_kind),
            "by_candidate_count": finish(by_count), "mismatch_examples": examples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/njupt")
    parser.add_argument("--probe", type=Path, default=ROOT / "bin/core_probe_rules")
    parser.add_argument("--train-limit", type=int, default=0,
                        help="optional deterministic cap; zero means all train rows")
    parser.add_argument("--validation-limit", type=int, default=0,
                        help="optional deterministic cap; zero means all validation rows")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/rule_distill_audit.json")
    args = parser.parse_args()
    if args.train_limit < 0 or args.validation_limit < 0:
        parser.error("limits must be nonnegative")
    source = args.source.resolve()
    probe_path = args.probe.resolve()
    # Explicitly enumerate only the two development files.  This guard is
    # intentional: future refactors must not silently open test.jsonl.
    split_paths = {"train": source / "train.jsonl", "validation": source / "validation.jsonl"}
    if any(not path.is_file() for path in split_paths.values()):
        raise FileNotFoundError("train.jsonl and validation.jsonl are required")
    limits = {"train": args.train_limit, "validation": args.validation_limit}
    with Probe(probe_path, timeout=20) as probe:
        splits = {name: audit_split(path, probe, limits[name]) for name, path in split_paths.items()}
    report = {"schema": "oxbot-rule-distill-audit-v1", "status": "complete",
              "scope": "RulePolicy pseudo-label agreement on train/validation only",
              "test_data_opened": False, "weights_changed": False,
              "source": str(source), "probe": str(probe_path),
              "probe_sha256": sha256_file(probe_path), "splits": splits}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value["totals"] for name, value in splits.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
