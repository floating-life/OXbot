# -*- coding: utf-8 -*-
"""Run the actual upload zip from a clean cwd with Botzone-like data/ storage.

Both traditional (fresh process per turn) and keep-running protocols must
complete official-judge games with the exact packed weight version loaded.
This is a local protocol/legality gate, not proof of Botzone CPU/RSS limits.
"""

import argparse
import ast
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile

from judge_runner import (JudgeHost, KeepProcBot, TradProcBot, make_initdata,
                          percentile, run_game)


def check(zip_path, weights, judge_path, games=1, timeout=60, report_path=None):
    zip_path = os.path.abspath(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        expected = archive.read("weights_name.txt").decode("utf-8").strip()
        for name in archive.namelist():
            if name.endswith(".py"):
                ast.parse(archive.read(name).decode("utf-8"), filename=name,
                          feature_version=(3, 6))
    with open(weights, "rb") as source:
        weight_sha = hashlib.sha256(source.read()).hexdigest()
    if expected != "fabledan_w_%s.npz" % weight_sha[:8]:
        raise ValueError("weights do not match weights_name.txt: %s" % expected)
    judge = JudgeHost(judge_path)
    rng = random.Random(20261003)
    deals = [make_initdata(rng, "ladder")[0] for _ in range(games)]
    report = {"zip": zip_path, "weight_sha256": weight_sha, "weight_name": expected,
              "judge": os.path.abspath(judge_path), "modes": {}, "errors": 0}
    with open(judge_path, "rb") as source:
        report["judge_sha256"] = hashlib.sha256(source.read()).hexdigest()
    # One BLAS thread per bot keeps local protocol tests from oversubscribing
    # an eight-core CPU. Environment changes apply only to child processes.
    with tempfile.TemporaryDirectory(prefix="oxbot_submission_") as storage:
        os.mkdir(os.path.join(storage, "data"))
        shutil.copyfile(weights, os.path.join(storage, "data", expected))
        for mode, driver in (("traditional", TradProcBot), ("keep_running", KeepProcBot)):
            times, results = [], []
            started = time.perf_counter()
            cmd = [sys.executable, zip_path]
            if mode == "keep_running":
                cmd.append("--keep-running")
            for initdata in deals:
                bots = [driver(cmd, cwd=storage, timeout=timeout, require_model=True)
                        for _ in range(4)]
                try:
                    result = run_game(judge, bots, initdata)
                finally:
                    for bot in bots:
                        times.extend(bot.times)
                        bot.close()
                results.append(result)
                if "error" in result:
                    report["errors"] += 1
                    break
            report["modes"][mode] = {
                "games": len(results), "results": results,
                "seconds": time.perf_counter() - started,
                "turn_seconds": {"p50": percentile(times, 50),
                                 "p99": percentile(times, 99),
                                 "max": max(times) if times else 0}}
            print("%s: %d game(s), %d errors, %.1fs" % (
                mode, len(results), sum("error" in result for result in results),
                report["modes"][mode]["seconds"]), flush=True)
    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
    if report["errors"]:
        raise RuntimeError("packed bot failed official-judge validation: %s" % report["modes"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--judge", default="judge/judge_official.py")
    parser.add_argument("--games", type=int, default=1, help="complete games per protocol")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--report", default="reports/packed_submission.json")
    args = parser.parse_args()
    if args.games < 1 or args.timeout <= 0:
        parser.error("games and timeout must be positive")
    check(args.zip, args.weights, args.judge, args.games, args.timeout, args.report)


if __name__ == "__main__":
    main()
