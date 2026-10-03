"""Mathematical checks for ambiguous-claim supervision and padded batches."""
from pathlib import Path
import math
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from train_bc import marginal_loss, weighted_marginal_loss, continuation_pairwise_loss


class BCObjectiveTests(unittest.TestCase):
    def test_probability_mass_of_multiple_valid_claims(self):
        scores = torch.tensor([[math.log(.2), math.log(.3), math.log(.5)]], requires_grad=True)
        positives = torch.tensor([[True, True, False]])
        loss = marginal_loss(scores, positives)
        self.assertAlmostEqual(float(loss.detach()), -math.log(.5), places=6)
        loss.backward()
        self.assertLess(float(scores.grad[0, 0]), 0)
        self.assertLess(float(scores.grad[0, 1]), 0)
        self.assertGreater(float(scores.grad[0, 2]), 0)

    def test_forced_action_has_no_training_signal(self):
        scores = torch.tensor([[2., float("-inf")]], requires_grad=True)
        loss = marginal_loss(scores, torch.tensor([[True, False]]))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(scores.grad, torch.zeros_like(scores)))

    def test_any_compatible_claim_is_sufficient(self):
        loss = marginal_loss(torch.tensor([[100., -100.]]), torch.tensor([[True, True]]))
        self.assertEqual(float(loss), 0.)

    def test_default_weighted_objective_is_unchanged(self):
        scores = torch.tensor([[0.2, -0.3], [1.1, -0.7]])
        positives = torch.tensor([[True, False], [False, True]])
        actions = torch.zeros((2, 2, 128))
        actions[0, 0, 120] = 0.2
        actions[1, 1, 120] = 0.3
        expected = marginal_loss(scores, positives)
        actual, _, _ = weighted_marginal_loss(scores, positives, actions, 1.0)
        self.assertAlmostEqual(float(actual), float(expected), places=7)

    def test_advantage_weights_are_row_normalized(self):
        scores = torch.tensor([[0.2, -0.3], [1.1, -0.7]])
        positives = torch.tensor([[True, False], [False, True]])
        actions = torch.zeros((2, 2, 128))
        row_weights = torch.tensor([1.25, 0.75])
        row_losses = marginal_loss(scores, positives, "none")
        expected = (row_losses * row_weights).sum() / row_weights.sum()
        actual, _, _ = weighted_marginal_loss(scores, positives, actions, 1.0, row_weights)
        self.assertAlmostEqual(float(actual), float(expected), places=7)

    def test_continuation_pairs_ignore_zero_and_rank_positive(self):
        scores = torch.tensor([[1.0, -1.0, 0.0]], requires_grad=True)
        labels = torch.tensor([[1, -1, 0]])
        mask = torch.tensor([[True, True, True]])
        loss, pairs, rows, correct = continuation_pairwise_loss(scores, labels, mask)
        self.assertEqual(pairs, 1)
        self.assertEqual(rows, 1)
        self.assertEqual(correct, 1)
        self.assertLess(float(loss), 0.2)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_continuation_pairwise_forced_rows_have_zero_loss(self):
        scores = torch.tensor([[2.0, -2.0]], requires_grad=True)
        labels = torch.tensor([[1, 0]])
        mask = torch.tensor([[True, True]])
        loss, pairs, rows, correct = continuation_pairwise_loss(scores, labels, mask)
        self.assertEqual((pairs, rows, correct), (0, 0, 0))
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
