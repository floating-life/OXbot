"""Describe train/validation demonstrations without opening test or tuning.

Outcome metadata is used only to describe the corpus, never as model input.
The leading comparison implements the attached referee's four-entry window.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from local_judge import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def referee_leading(history, player):
    leading = True
    for event in reversed(history[-4:]):
        if event["player"] == player:
            break
        if event["action"]:
            leading = False
            break
    return leading


def count_action(counts, size, leading):
    counts["decisions"] += 1
    counts["cards_played"] += size
    counts["pass" if size == 0 else "single" if size == 1 else "multi"] += 1
    if leading:
        counts["leading"] += 1
        counts["leading_pass" if size == 0 else "leading_single" if size == 1 else "leading_multi"] += 1


def diagnose(path):
    counts = collections.Counter()
    outcome_counts = collections.defaultdict(collections.Counter)
    match_counts = collections.defaultdict(collections.Counter)
    mismatch_examples = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["stage"] != "play":
            continue
        features = row["features"]
        leading = features["leading"]
        size = len(row["label"]["cards"])
        count_action(counts, size, leading)
        count_action(match_counts[row["match_id"]], size, leading)
        winner = row["result"]["single_game"]["winner_seat"]
        cohort = "eventual_winning_team" if winner % 2 == features["seat"] % 2 else "eventual_losing_team"
        count_action(outcome_counts[cohort], size, leading)
        if referee_leading(features["history"], features["seat"]) != leading:
            counts["leading_mismatches"] += 1
            if len(mismatch_examples) < 12:
                mismatch_examples.append({key: row[key] for key in ("game_id", "deal_id", "event_index")} |
                                         {"features_leading": leading, "player": features["seat"],
                                          "history_last_four": features["history"][-4:]})
    return {"path": path.as_posix(), "sha256": sha256_file(path), "counts": dict(counts),
            "outcome_cohorts": {key: dict(value) for key, value in sorted(outcome_counts.items())},
            "match_cohorts": {key: dict(value) for key, value in sorted(match_counts.items())},
            "leading_mismatch_examples": mismatch_examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/njupt")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/njupt_behavior_diagnostic.json")
    args = parser.parse_args()
    result = {"scope": "descriptive audit of train and validation only; no filtering or tuning",
              "test_data_opened": False, "weights_changed": False,
              "oracle_sha256": sha256_file(ROOT / "裁判代码-修正版.py"),
              "leading_definition": "exact referee reverse scan of four most recent plays; stop at self or nonpass",
              "splits": {split: diagnose(args.source / f"{split}.jsonl") for split in ("train", "validation")}}
    atomic_json(args.report, result)
    print(json.dumps({split: {key: audit[key] for key in ("counts", "outcome_cohorts", "leading_mismatch_examples")}
                      for split, audit in result["splits"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
