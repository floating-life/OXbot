# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cpp_duel  # noqa: E402


def shard(records, errors=0, play=10, model=10):
    return {"errors": errors, "game_records": records,
            "turn_seconds_by_team": {"a": {"n": 5, "p50": .01, "p99": .05, "max": .2},
                                     "b": {"n": 5, "p50": .02, "p99": .06, "max": .3}},
            "cpp_model": {"play_turns": play, "model_selected_turns": model,
                          "model_sha_prefixes": ["aaaaaaaaaaaa"]}}


def games(pair, first, second):
    return [{"pair": pair, "a_seats": [0, 2], "a_diff": first},
            {"pair": pair, "a_seats": [1, 3], "a_diff": second}]


class MergeTest(unittest.TestCase):
    def test_complete_shards_merge_by_deal(self):
        reports = [shard(games(0, 3, -1) + games(2, 2, 1)), shard(games(1, 1, 1) + games(3, 3, 2))]
        s = cpp_duel.merge(reports, 4, iters=500)
        self.assertTrue(s["complete"])
        self.assertEqual(s["deals"], 4)
        self.assertAlmostEqual(s["avg_score_diff_per_game"], 12 / 8)
        self.assertEqual(s["turn_seconds"]["b"]["max"], .3)

    def test_missing_game_or_fallback_is_invalid(self):
        partial = cpp_duel.merge([shard(games(0, 3, 3) + games(1, 3, 3)[:1])], 2, iters=100)
        self.assertEqual(partial["verdict"], "INVALID")
        fallback = cpp_duel.merge([shard(games(0, 3, 3) + games(1, 3, 3), model=9)], 2, iters=100)
        self.assertEqual(fallback["verdict"], "INVALID")
        judged = cpp_duel.merge([shard(games(0, 3, 3) + games(1, 3, 3), errors=1)], 2, iters=100)
        self.assertEqual(judged["verdict"], "INVALID")


class JudgeRunnerShardTest(unittest.TestCase):
    def run_runner(self, folder, name, extra):
        path = os.path.join(folder, name)
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "judge_runner.py"),
                        "--games", "8", "--seed", "3", "--weights", "rule",
                        "--weights-b", "rule", "--report", path] + extra,
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        with open(path) as f:
            return json.load(f)

    def test_shards_partition_the_same_schedule(self):
        with tempfile.TemporaryDirectory() as folder:
            whole = self.run_runner(folder, "all.json", [])
            parts = [self.run_runner(folder, "s%d.json" % k, ["--shard", "%d/2" % k])
                     for k in range(2)]
        key = lambda r: (r["pair"], tuple(r["a_seats"]))
        merged = sorted((rec for p in parts for rec in p["game_records"]), key=key)
        self.assertEqual(merged, sorted(whole["game_records"], key=key))
        self.assertEqual([p["games"] for p in parts], [4, 4])
        self.assertTrue({r["pair"] for r in parts[0]["game_records"]}.isdisjoint(
            {r["pair"] for r in parts[1]["game_records"]}))


if __name__ == "__main__":
    unittest.main()
