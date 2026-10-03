"""Summarize fixed-seed raw vs group-logmeanexp duplicate reports.

This is a descriptive A/B guardrail only.  It validates complementary-seat
duplicate keys and reports legality/fallback/finish proxies; it makes no
strength or confidence claim and never opens training shards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed":
        raise ValueError(f"{path}: report is not passed")
    return value


def by_key(report):
    records = report.get("game_records")
    if not isinstance(records, list):
        raise ValueError("game_records missing")
    result = {}
    for record in records:
        key = record["deal_key"]
        if key in result:
            raise ValueError(f"duplicate deal key: {key}")
        result[key] = record
    return result


def paired(left, right):
    one, two = by_key(left), by_key(right)
    keys = sorted(set(one) & set(two))
    if len(keys) != len(one) or len(keys) != len(two):
        raise ValueError("complementary reports do not cover the same deals")
    margins = []
    for key in keys:
        a, b = one[key], two[key]
        if {str(a["model_team"]), str(b["model_team"])} != {"0", "1"}:
            raise ValueError("duplicate pair is not complementary")
        first_sign = 1 if str(a["model_team"]) == "0" else -1
        second_sign = 1 if str(b["model_team"]) == "0" else -1
        margins.append(first_sign * (a["team_scores"][0] - a["team_scores"][1]) +
                       second_sign * (b["team_scores"][0] - b["team_scores"][1]))
    return {"pairs": len(margins), "positive_pairs": sum(v > 0 for v in margins),
            "tied_pairs": sum(v == 0 for v in margins), "negative_pairs": sum(v < 0 for v in margins),
            "total_model_point_margin": sum(margins),
            "mean_model_point_margin_per_pair": sum(margins) / len(margins) if margins else None,
            "inference": "descriptive paired points only; no strength threshold or confidence claim"}


def metrics(report):
    games = report["game_records"]
    return {"games": len(games), "decisions": report["counts"].get("decisions", 0),
            "play_decisions": report["stages"].get("play", 0),
            "fallbacks": report["fallback_counts"], "policy_counts": report["policies"],
            "play_policy_counts": report["play_policies"],
            "model_status_counts": report["model_status_counts"],
            "budget_exceedances": report["local_wall_clock_budget_exceedances"],
            "winning_teams": report["winning_teams"],
            "turns_total": sum(item["turns"] for item in games),
            "turns_min": min((item["turns"] for item in games), default=None),
            "turns_max": max((item["turns"] for item in games), default=None)}


def main():
    parser = argparse.ArgumentParser()
    for name in ("raw0", "raw1", "group0", "group1"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {name: load(getattr(args, name)) for name in ("raw0", "raw1", "group0", "group1")}
    for name, report in reports.items():
        if report.get("seed") != reports["raw0"].get("seed") or report.get("requested_games") != reports["raw0"].get("requested_games"):
            raise ValueError(f"{name}: seed/game count mismatch")
    for name in ("raw0", "raw1"):
        if reports[name].get("selection_strategy") != "raw":
            raise ValueError(f"{name}: expected raw strategy")
    for name in ("group0", "group1"):
        if reports[name].get("selection_strategy") != "group-logmeanexp":
            raise ValueError(f"{name}: expected group-logmeanexp strategy")
    result = {
        "schema": "oxbot-selection-ab-v1", "status": "complete",
        "scope": "fixed-seed duplicate A/B; attachment-referee legality and local guardrails only",
        "test_split_read": False, "weights_changed": False,
        "model": reports["raw0"].get("model"),
        "oracle_sha256": reports["raw0"].get("oracle_sha256"),
        "seed": reports["raw0"].get("seed"), "groups": reports["raw0"].get("requested_games"),
        "strategies": {
            "raw": {"team0": metrics(reports["raw0"]), "team1": metrics(reports["raw1"]),
                    "paired": paired(reports["raw0"], reports["raw1"])},
            "group-logmeanexp": {"team0": metrics(reports["group0"]), "team1": metrics(reports["group1"]),
                                  "paired": paired(reports["group0"], reports["group1"])},
        },
        "reports": {name: str(getattr(args, name).resolve()) for name in ("raw0", "raw1", "group0", "group1")},
        "guardrail": {"all_reports_passed": True, "all_games_completed": True,
                       "model_legality_fallbacks": sum(report["fallback_counts"].get("legality_recheck_fallback", 0)
                                                        for report in reports.values()),
                       "model_required_fallbacks": sum(report["fallback_counts"].get("model_requested_but_not_used", 0)
                                                       for report in reports.values()),
                       "local_budget_exceedances": sum(sum(report["local_wall_clock_budget_exceedances"].values())
                                                       for report in reports.values())},
        "recommendation": "do_not_replace_raw_without_larger strength-controlled evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
