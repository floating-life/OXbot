import unittest

from train.build_candidate_credit_targets import _candidate_labels


class FakeProbe:
    def __init__(self, accepted_actions):
        self.accepted_actions = {tuple(action) for action in accepted_actions}

    def call(self, *, command, previous, move, level):
        self.assert_command = command
        return {"ok": tuple(previous[0]) in self.accepted_actions}


def record(leading=False):
    return {"features": {"own_hand": [0, 1, 2, 3, 4, 5],
                          "leading": leading, "level_label": "2"}}


class CandidateContinuationTargetTests(unittest.TestCase):
    def test_pair_labels_mean_compatibility_not_next_team(self):
        moves = [[[1], [1]], [[3], [3]]]
        metadata = [{"kind": "single", "key": 1, "secondary": -1},
                    {"kind": "single", "key": 3, "secondary": -1}]
        support = {"status": "supported", "current_previous": [[0], [0]],
                   "first_future": [[2], [2]], "future_actor_faces": {}}
        labels, summary = _candidate_labels(record(), moves, metadata, support,
                                             FakeProbe(accepted_actions=([1],)))
        self.assertEqual(labels, [1, -1])
        self.assertEqual(summary["positive"], 1)
        self.assertEqual(summary["negative"], 1)
        self.assertTrue(summary["pair"])

    def test_pass_uses_existing_previous_comparison(self):
        moves = [[[], []], [[3], [3]]]
        metadata = [{"kind": "pass", "key": 0, "secondary": 0},
                    {"kind": "single", "key": 3, "secondary": -1}]
        support = {"status": "supported", "current_previous": [[1], [1]],
                   "first_future": [[2], [2]], "future_actor_faces": {}}
        labels, summary = _candidate_labels(record(), moves, metadata, support,
                                             FakeProbe(accepted_actions=([1],)))
        self.assertEqual(labels, [1, -1])
        self.assertTrue(summary["pair"])

    def test_unsupported_rows_are_ignored(self):
        moves = [[[1], [1]], [[3], [3]]]
        metadata = [{"kind": "single", "key": 1, "secondary": -1},
                    {"kind": "single", "key": 3, "secondary": -1}]
        labels, summary = _candidate_labels(
            record(), moves, metadata, {"status": "unsupported"},
            FakeProbe(accepted_actions=([1],)))
        self.assertEqual(labels, [0, 0])
        self.assertEqual(summary["labeled"], 0)
        self.assertFalse(summary["pair"])


if __name__ == "__main__":
    unittest.main()
