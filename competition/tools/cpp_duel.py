# -*- coding: utf-8 -*-
"""Local C++ vs C++ duplicate match through the official judge.

Replaces BotZone ladder matches for strength checks: both bots run as
long-running C++ processes (exactly the protocol BotZone uses), every deal is
played twice with the teams swapped, and the deal is the statistical unit.
Work is split into shards that run in parallel; every shard draws the same
deal schedule, so the result does not depend on the number of workers.

    python tools/cpp_duel.py \
        --bot-a "../bin/oxbot --model ../data/fabledan_w_xxxx.fbd" \
        --bot-b "../bin/oxbot --model ../models/cf8.bin" \
        --deals 500 --workers 14 --report ../reports/duel/new_vs_cf8.json

Verdict A_STRONGER / A_WEAKER / INCONCLUSIVE comes from the paired-bootstrap
95% CI of A's score difference per game; any judge error or model fallback
makes the run INVALID.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from duplicate_eval import describe, summarize  # noqa: E402


def merge(shard_reports, deals, iters=10000, seed=0):
    """Combine shard reports into one duplicate summary."""
    errors = sum(r["errors"] for r in shard_reports)
    games = {}
    for report in shard_reports:
        for record in report.get("game_records", []):
            games[(record["pair"], tuple(record["a_seats"]))] = record["a_diff"]
    pairs = []
    for pair in range(deals):
        first = games.get((pair, (0, 2)))
        second = games.get((pair, (1, 3)))
        if first is not None and second is not None:
            pairs.append((float(first), float(second)))
    summary = summarize(pairs, iters, seed) if pairs else {"deals": 0, "verdict": "INVALID"}
    timing = {}
    for team in ("a", "b"):
        rows = [r.get("turn_seconds_by_team", {}).get(team) for r in shard_reports]
        rows = [row for row in rows if row and row.get("n")]
        if rows:
            timing[team] = {
                "turns": sum(row["n"] for row in rows),
                # Shard percentiles cannot be combined exactly; report the
                # worst shard, which is the conservative reading for time limits.
                "p50_max_shard": max(row["p50"] for row in rows),
                "p99_max_shard": max(row["p99"] for row in rows),
                "max": max(row["max"] for row in rows),
            }
    shas = sorted({sha for r in shard_reports
                   for sha in r.get("cpp_model", {}).get("model_sha_prefixes", [])})
    play_turns = sum(r.get("cpp_model", {}).get("play_turns", 0) for r in shard_reports)
    model_turns = sum(r.get("cpp_model", {}).get("model_selected_turns", 0) for r in shard_reports)
    complete = len(pairs) == deals and errors == 0 and play_turns == model_turns
    if not complete:
        summary["verdict"] = "INVALID"
    summary.update(judge_errors=errors, requested_deals=deals, complete=complete,
                   play_turns=play_turns, model_selected_turns=model_turns,
                   model_sha_prefixes=shas, turn_seconds=timing)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot-a", required=True, help="command line of the candidate C++ bot")
    ap.add_argument("--bot-b", required=True, help="command line of the opponent C++ bot")
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--scenario", default="mix",
                    help="judge_runner scenario (mix = all tribute/level settings)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--judge", default=os.path.join(ROOT, "judge", "judge_official.py"))
    ap.add_argument("--timeout", type=float, default=10.0, help="per-turn local timeout (s)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--artifact-a", default=None, help="file to fingerprint for bot A (weights)")
    ap.add_argument("--artifact-b", default=None, help="file to fingerprint for bot B (weights)")
    args = ap.parse_args()
    if args.deals < 2 or args.workers < 1:
        ap.error("--deals must be >= 2 and --workers >= 1")
    workers = min(args.workers, args.deals)

    report_dir = os.path.dirname(os.path.abspath(args.report))
    os.makedirs(report_dir, exist_ok=True)
    shard_dir = tempfile.mkdtemp(prefix="shards-", dir=report_dir)
    t0 = time.time()
    procs = []
    for k in range(workers):
        path = os.path.join(shard_dir, "shard%02d.json" % k)
        cmd = [sys.executable, os.path.join(HERE, "judge_runner.py"),
               "--judge", args.judge, "--games", str(2 * args.deals),
               "--seed", str(args.seed), "--scenario", args.scenario,
               "--driver", "cpp:" + args.bot_a, "--driver-b", "cpp:" + args.bot_b,
               "--require-model", "--timeout", str(args.timeout),
               "--shard", "%d/%d" % (k, workers), "--report", path]
        log = open(os.path.join(shard_dir, "shard%02d.log" % k), "w")
        procs.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), path, log))
    shard_reports, failed = [], []
    for proc, path, log in procs:
        proc.wait()
        log.close()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                shard_reports.append(json.load(f))
        else:
            failed.append(path)
    summary = merge(shard_reports, args.deals, seed=args.seed)
    if failed:
        summary["verdict"] = "INVALID"
        summary["complete"] = False
        summary["missing_shards"] = failed
    report = {"bot_a": args.bot_a, "bot_b": args.bot_b,
              "artifact_a": describe(args.artifact_a) if args.artifact_a else None,
              "artifact_b": describe(args.artifact_b) if args.artifact_b else None,
              "seed": args.seed, "scenario": args.scenario, "workers": workers,
              "judge": os.path.abspath(args.judge),
              "elapsed_seconds": time.time() - t0, "shard_dir": shard_dir, **summary}
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    ci = summary.get("ci95_per_game", [float("nan")] * 2)
    print("A vs B: %+.3f/game, 95%% CI [%+.3f, %+.3f], %d/%d deals, judge errors %d -> %s"
          % (summary.get("avg_score_diff_per_game", float("nan")), ci[0], ci[1],
             summary.get("deals", 0), args.deals, summary["judge_errors"], summary["verdict"]))
    for team, row in summary["turn_seconds"].items():
        print("  bot %s: p99 %.0f ms (worst shard), max %.0f ms"
              % (team.upper(), 1000 * row["p99_max_shard"], 1000 * row["max"]))
    sys.exit(0 if summary["verdict"] != "INVALID" else 4)


if __name__ == "__main__":
    main()
