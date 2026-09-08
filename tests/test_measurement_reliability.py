"""State slice: aligned-holistic-continual-learning-integrated-stack-v2."""
import importlib.util
import math
import unittest

from ahcl.checks import CheckError, assert_parity, assert_intervention_reach, causal_scrub_effect, normalized_reconstruction_mse


class QualificationChecks(unittest.TestCase):
    def test_nonfinite_at_any_position_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(CheckError):
                assert_parity([0, bad], [0, 0])

    def test_equal_element_count_is_not_shape_parity(self):
        with self.assertRaises(CheckError):
            assert_parity([[1, 2]], [[1], [2]])

    def test_empty_and_ragged_inputs_rejected(self):
        for value in ([], [[1], [2, 3]]):
            with self.assertRaises(CheckError):
                assert_parity(value, value)

    def test_invalid_thresholds_rejected(self):
        for threshold in (-1, math.nan, math.inf):
            with self.assertRaises(CheckError):
                assert_parity([0], [0], atol=threshold)
        with self.assertRaises(CheckError):
            assert_intervention_reach([0], [0], minimum_delta=0)

    def test_reconstruction_and_scrub_reject_invalid_inputs(self):
        with self.assertRaises(CheckError):
            normalized_reconstruction_mse([[1, 2]], [1, 2])
        with self.assertRaises(CheckError):
            causal_scrub_effect([0, math.nan], [0, 1])
        self.assertAlmostEqual(normalized_reconstruction_mse([1, 2], [0, 2]), 0.2)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch runtime not installed; tensor qualification pending")
class NextTokenChecks(unittest.TestCase):
    def test_shift_mask_and_policy_denominator(self):
        import torch
        from ahcl.evaluators import align_next_tokens, exact_match_accuracy, policy_violation_rate
        logits = torch.tensor([[[0., 9., 0.], [0., 0., 9.], [9., 0., 0.], [9., 0., 0.]]])
        labels = torch.tensor([[0, 1, 2, 0]])
        aligned, targets = align_next_tokens(logits, labels, torch.tensor([[1, 1, 1, 0]]))
        self.assertEqual(targets.tolist(), [[1, 2, -100]])
        self.assertEqual(exact_match_accuracy(aligned, targets), 1.)
        self.assertEqual(policy_violation_rate(aligned, [0], labels=targets), 0.)

    def test_left_padding_and_empty_scoring(self):
        import torch
        from ahcl.evaluators import align_next_tokens
        logits = torch.zeros((1, 3, 3))
        labels = torch.tensor([[0, 1, 2]])
        _, targets = align_next_tokens(logits, labels, torch.tensor([[0, 1, 1]]))
        self.assertEqual(targets.tolist(), [[-100, 2]])
        with self.assertRaises(ValueError):
            align_next_tokens(logits, labels, torch.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main()
