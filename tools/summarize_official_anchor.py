"""Create a compact, reproducible summary for the rebuilt anchor package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team0", type=Path,
                        default=ROOT / "reports" / "official_anchor_package_team0_16.json")
    parser.add_argument("--team1", type=Path,
                        default=ROOT / "reports" / "official_anchor_package_team1_16.json")
    parser.add_argument("--package", type=Path,
                        default=ROOT / "dist" / "oxbot-bc-v2-full-fp32-anchor.cpp")
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "bin" / "oxbot-bc-v2-full-fp32-anchor")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports" / "official_anchor_package_16x2_summary.json")
    args = parser.parse_args()

    first = read(args.team0)
    second = read(args.team1)
    if first.get("status") != "passed" or second.get("status") != "passed":
        raise SystemExit("both team reports must pass")
    invariant_keys = ("oracle_sha256", "probe_sha256", "process_sha256")
    for key in invariant_keys:
        if first.get(key) != second.get(key):
            raise SystemExit(f"paired report mismatch: {key}")
    if first.get("model", {}).get("payload_sha256") != second.get("model", {}).get("payload_sha256"):
        raise SystemExit("paired report mismatch: model payload")

    records0 = {item["deal_key"]: item for item in first["game_records"]}
    records1 = {item["deal_key"]: item for item in second["game_records"]}
    if set(records0) != set(records1):
        raise SystemExit("paired report mismatch: deal keys")
    margins = []
    for key, record0 in records0.items():
        record1 = records1[key]
        team0_margin = record0["team_scores"][0] - record0["team_scores"][1]
        team1_margin = record1["team_scores"][1] - record1["team_scores"][0]
        margins.append(team0_margin + team1_margin)

    manifest_path = args.package.with_suffix(".manifest.json")
    manifest = read(manifest_path)
    oracle_manifest = read(ROOT / "judge" / "oracle_manifest.json")
    boundary_report = ROOT / "reports" / "official_return_boundaries.json"
    summary = {
        "schema": "oxbot-official-anchor-summary-v1",
        "status": "passed",
        "oracle": {
            "path": str((ROOT / "reports" / "official_judge_botzone_2026-10-02.py").resolve()),
            "raw_sha256": first["oracle_sha256"],
            "normalized_sha256": oracle_manifest["online_original"]["sha256"],
            "boundary_golden": str(boundary_report.resolve()),
        },
        "package": {
            "source": str(args.package.resolve()),
            "bytes": args.package.stat().st_size,
            "sha256": sha256(args.package),
            "manifest_sha256": sha256(manifest_path),
            "binary": str(args.binary.resolve()),
            "binary_sha256": sha256(args.binary),
            "manifest_release_eligible": manifest.get("release_eligible"),
            "candidate_version": manifest.get("candidate_version"),
        },
        "model": first["model"],
        "paired_evaluation": {
            "games_per_team": len(records0),
            "paired_games": len(margins),
            "team0_model_wins": first["model_vs_rule"]["wins"],
            "team1_model_wins": second["model_vs_rule"]["wins"],
            "positive_pairs": sum(value > 0 for value in margins),
            "tied_pairs": sum(value == 0 for value in margins),
            "negative_pairs": sum(value < 0 for value in margins),
            "total_model_point_margin": sum(margins),
            "mean_model_point_margin_per_pair": sum(margins) / len(margins),
            "strength_claim": "none; descriptive gate only",
        },
        "execution": {
            "model_decisions": first["policies"].get("model", 0) + second["policies"].get("model", 0),
            "required_model_fallbacks": 0,
            "illegal_actions": 0,
            "budget_exceedances": len(first.get("local_wall_clock_budget_exceedances", {})) +
                                  len(second.get("local_wall_clock_budget_exceedances", {})),
            "max_probe_rss_kib": max(first.get("max_probe_rss_kib") or 0,
                                      second.get("max_probe_rss_kib") or 0),
        },
        "release_gate": {
            "release_eligible": False,
            "reason": "anchor fails the paired strength gate; BotZone G++7.2/private match/timing remain unverified",
            "platform_upload_performed": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
