# -*- coding: utf-8 -*-
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import duplicate_eval as D  # noqa: E402


class DuplicateEvalTest(unittest.TestCase):
    def test_summary_uses_deals_as_unit(self):
        pairs = [(3.0, -1.0)] * 30 + [(-1.0, 1.0)] * 10
        s = D.summarize(pairs, iters=2000, seed=1)
        self.assertEqual(s["deals"], 40)
        self.assertEqual(s["games"], 80)
        self.assertEqual((s["deal_wins"], s["deal_ties"], s["deal_losses"]), (30, 10, 0))
        self.assertAlmostEqual(s["avg_score_diff_per_game"], 0.75)
        self.assertEqual(s["verdict"], "A_STRONGER")

    def test_symmetric_results_are_inconclusive(self):
        pairs = [(1.0, 1.0), (-1.0, -1.0)] * 20
        s = D.summarize(pairs, iters=2000, seed=1)
        self.assertEqual(s["verdict"], "INCONCLUSIVE")
        lo, hi = s["ci95_per_game"]
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_rule_beats_random(self):
        D._init_worker("rule", "random", 7)
        pairs = [D.play_deal((i, 1000 + i, 0.5))[1] for i in range(12)]
        self.assertEqual(D.summarize(pairs, iters=500)["verdict"], "A_STRONGER")


if __name__ == "__main__":
    unittest.main()
