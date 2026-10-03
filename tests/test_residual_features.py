import unittest

from train.residual_features import (
    RESIDUAL_FEATURE_VERSION,
    residual_features_from_summaries,
    residual_hand,
    summarize_lead_moves,
)


class ResidualFeatureTests(unittest.TestCase):
    def test_physical_card_removal_is_exact(self):
        self.assertEqual(residual_hand([0, 54, 1, 2], [54, 1]), [0, 2])
        with self.assertRaises(ValueError):
            residual_hand([0, 1], [0, 0])

    def test_summary_counts_only_nonpass_leads_and_power(self):
        moves = [
            [[0], [0]],
            [[1, 2], [1, 2]],
            [[3, 4, 5, 6], [3, 4, 5, 6]],
            [[7, 8, 9, 10, 11], [7, 8, 9, 10, 11]],
        ]
        types = [
            {"kind": "single"}, {"kind": "pair"},
            {"kind": "bomb"}, {"kind": "straight_flush"},
        ]
        summary = summarize_lead_moves(moves, types)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["power"], 2)
        self.assertEqual(summary["bomb_max_length"], 4)
        self.assertEqual(summary["by_kind"]["straight_flush"], 1)
        with self.assertRaises(ValueError):
            summarize_lead_moves([[], []], [{"kind": "pass"}])

    def test_local_feature_has_no_opponent_or_result_fields(self):
        base = {"total": 8}
        base = {**base, "power": 2, "nonpower": 6}
        residual = {
            "total": 3, "power": 1, "nonpower": 2,
            "bomb_max_length": 4, "has_rocket": True,
            "by_kind": {"single": 2, "bomb": 1},
        }
        row = residual_features_from_summaries([0, 1, 2, 3, 4, 5], [0, 1], "2", base, residual)
        self.assertEqual(row["schema"], RESIDUAL_FEATURE_VERSION)
        self.assertEqual(row["residual_hand_size"], 4)
        self.assertEqual(row["candidate_action_size"], 2)
        self.assertEqual(row["residual_lead_total"], 3)
        self.assertEqual(row["lead_total_delta"], -5)
        self.assertNotIn("result", row)
        self.assertNotIn("opponent_hand", row)


if __name__ == "__main__":
    unittest.main()
