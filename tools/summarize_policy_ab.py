"""Summarize a fixed-seed complementary-seat policy A/B pair.

This is a descriptive local guardrail only.  It reads completed local-judge
reports, verifies that the two reports cover the same deals and that the
strategy/debug identity is consistent, then computes paired point margins.
It never reads training or held-out test data and makes no strength claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path, strategy: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed":
        raise ValueError(f"{path}: local report is not passed")
    if value.get("selection_strategy") != strategy:
        raise ValueError(f"{path}: expected strategy {strategy!r}")
    if value.get("selection_strategy_version") != strategy + "-v1":
        raise ValueError(f"{path}: strategy version mismatch")
    return value


def records(report: dict) -> dict[str, dict]:
    output = {}
    for record in report.get("game_records", []):
        key = record["deal_key"]
        if key in output:
            raise ValueError(f"duplicate deal key {key}")
        output[key] = record
    return output


def paired(left: dict, right: dict) -> dict:
    one, two = records(left), records(right)
    if set(one) != set(two):
        raise ValueError("complementary reports do not cover identical deals")
    margins = []
    for key in sorted(one):
        a, b = one[key], two[key]
        if {str(a["model_team"]), str(b["model_team"])} != {"0", "1"}:
            raise ValueError(f"deal {key} is not a complementary seat pair")
        # Convert each leg to the model team's point margin before adding.
        first = 1 if str(a["model_team"]) == "0" else -1
        second = 1 if str(b["model_team"]) == "0" else -1
        margins.append(first * (a["team_scores"][0] - a["team_scores"][1]) +
                       second * (b["team_scores"][0] - b["team_scores"][1]))
    return {"pairs": len(margins), "positive_pairs": sum(v > 0 for v in margins),
            "tied_pairs": sum(v == 0 for v in margins), "negative_pairs": sum(v < 0 for v in margins),
            "total_model_point_margin": sum(margins),
            "mean_model_point_margin_per_pair": sum(margins) / len(margins) if margins else None,
            "inference": "descriptive paired points only; no strength threshold or confidence claim"}


def compact(report: dict) -> dict:
    return {"games": len(report.get("game_records", [])),
            "decisions": report.get("counts", {}).get("decisions", 0),
            "play_decisions": report.get("stages", {}).get("play", 0),
            "fallback_counts": report.get("fallback_counts", {}),
            "winning_teams": report.get("winning_teams", {}),
            "local_budget_exceedances": report.get("local_wall_clock_budget_exceedances", {}),
            "model_required": report.get("require_model"),
            "model_play": report.get("play_policies", {}).get("model", 0),
            "rule_fallback_play": report.get("play_policies", {}).get("rule_fallback", 0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team0", type=Path, required=True)
    parser.add_argument("--team1", type=Path, required=True)
    parser.add_argument("--strategy", choices=("raw", "raw-pass-bias", "group-logmeanexp"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left, right = load(args.team0, args.strategy), load(args.team1, args.strategy)
    for name, report in (("team0", left), ("team1", right)):
        if report.get("seed") != left.get("seed") or report.get("requested_games") != left.get("requested_games"):
            raise ValueError(f"{name}: seed/game count mismatch")
        if report.get("oracle_sha256") != left.get("oracle_sha256"):
            raise ValueError(f"{name}: oracle mismatch")
        if report.get("model", {}).get("payload_sha256") != left.get("model", {}).get("payload_sha256"):
            raise ValueError(f"{name}: model mismatch")
    result = {"schema": "oxbot-policy-ab-v1", "status": "complete",
              "scope": "fixed-seed complementary-seat policy A/B; attachment-referee local guardrail only",
              "test_split_read": False, "weights_changed": False, "strategy": args.strategy,
              "seed": left.get("seed"), "groups": left.get("requested_games"),
              "model": left.get("model"), "oracle_sha256": left.get("oracle_sha256"),
              "team0": compact(left), "team1": compact(right), "paired": paired(left, right),
              "reports": {"team0": str(args.team0.resolve()), "team1": str(args.team1.resolve())},
              "guardrail": {"reports_passed": True, "same_deals": True,
                             "model_required_fallbacks": sum(r.get("fallback_counts", {}).get("model_requested_but_not_used", 0)
                                                              for r in (left, right)),
                             "legality_recheck_fallbacks": sum(r.get("fallback_counts", {}).get("legality_recheck_fallback", 0)
                                                                for r in (left, right)),
                             "local_budget_exceedances": sum(sum(r.get("local_wall_clock_budget_exceedances", {}).values())
                                                             for r in (left, right))},
              "recommendation": "descriptive only; retain raw default until a stronger model passes larger controlled evaluation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
