import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from features import history_tokens, matching_actions, state_features


class FeatureTests(unittest.TestCase):
    def test_other_private_hands_cannot_change_information_set(self):
        observation = {"hand": [4, 58, 8], "level": "2", "player": 0, "leading": True,
                       "remaining_counts": [3, 27, 27, 27], "history": []}
        leaked = copy.deepcopy(observation)
        leaked["opponent_hands"] = [[107], [3], [18]]
        self.assertTrue((state_features(observation) == state_features(leaked)).all())
        self.assertEqual(history_tokens(observation).tolist(), [1, 118])

    def test_seat_rotation_invariance(self):
        obs = {"hand": [3], "level": "10", "player": 1, "leading": False,
               "remaining_counts": [20, 1, 21, 22],
               "history": [{"player": 0, "action": [8], "claim": [8]}]}
        rotated = copy.deepcopy(obs)
        rotated["player"] = 3
        rotated["history"][0]["player"] = 2
        rotated["remaining_counts"] = [21, 22, 20, 1]
        self.assertTrue((state_features(obs) == state_features(rotated)).all())
        self.assertTrue((history_tokens(obs) == history_tokens(rotated)).all())

    def test_ambiguous_claims_and_deck_copies(self):
        moves = [[[4, 58], [0, 0]], [[4, 58], [8, 8]], [[8], [8]]]
        self.assertEqual(matching_actions(moves, [58, 4]), [0, 1])
        self.assertEqual(matching_actions([[[8], [8]]], [62]), [0])


if __name__ == "__main__":
    unittest.main()
