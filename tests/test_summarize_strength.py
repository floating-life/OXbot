from __future__ import annotations
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from summarize_strength import audited_pairs, paired_bootstrap, summarize


def report(team):
    games = [{"index": i, "seed": 20262001 + i, "level": "2", "tribute": 0, "first": 0, "last": 1,
              "deal_key": f"{i:064x}", "model_team": str(team), "team_scores": [3, 0], "winner_team": 0} for i in range(2)]
    return {"schema": "oxbot-local-judge-v2", "status": "passed", "failure": None, "require_model": True,
            "model_team": str(team), "oracle_sha256": "a" * 64, "probe_sha256": "b" * 64,
            "model": {"file_sha256": "c" * 64, "payload_sha256": "d" * 64}, "play_policies": {"model": 20},
            "observed_model_payload_sha_prefix_counts": {"d" * 12: 20}, "requested_games": 2, "counts": {"games": 2},
            "seed": 20262001, "mode": "offline_probe", "game_records": games}


class DuplicateStrengthTests(unittest.TestCase):
    def pair(self):
        first, second = report(0), report(1)
        second["paired_scores"] = {"pairs": 2, "unmatched_first": 0, "unmatched_second": 0, "total_model_point_margin": 0}
        return first, second

    def test_bootstrap_resamples_groups_without_breaking_the_legs(self):
        first, second = self.pair()
        result = summarize(first, second, expected_pairs=2, expected_seed=20262001, repetitions=1000)
        # The model wins one of the two legs of EVERY group. Resampling the
        # four legs independently would falsely produce a nonzero interval.
        self.assertEqual(result["metrics"]["model_win_fraction_per_game"]["ci95_percentile"], [.5, .5])
        self.assertEqual(result["metrics"]["model_point_margin_per_game"]["ci95_percentile"], [0., 0.])

    def test_pair_order_and_fixed_seed_are_reproducible(self):
        first, second = self.pair()
        a = audited_pairs(first, second)
        second["game_records"].reverse()
        b = audited_pairs(second, first)
        self.assertEqual(a, b)
        self.assertEqual(paired_bootstrap(a, 1000, 7), paired_bootstrap(b, 1000, 7))

    def test_identity_and_pairing_fail_closed(self):
        first, original_second = self.pair()
        for field in ("oracle_sha256", "probe_sha256", "seed"):
            second = copy.deepcopy(original_second)
            second[field] = "e" * 64 if field.endswith("sha256") else 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                audited_pairs(first, second)
        second = copy.deepcopy(original_second)
        second["game_records"][0]["deal_key"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "deal-key"):
            audited_pairs(first, second)
        second = copy.deepcopy(original_second)
        second["model"]["payload_sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            audited_pairs(first, second)


if __name__ == "__main__":
    unittest.main()
