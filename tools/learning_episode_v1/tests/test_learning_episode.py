"""Tests for the local synthetic receipt path.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
Governance slice: global-experiment-assurance-lifecycle-v1.
"""

from __future__ import annotations

import copy
import unittest

from ahcl.canonical import digest
from learning_episode import SealedAssessment, run_episode
from validate_learning_episode import ValidationError, validate_receipt


class LearningEpisodeTests(unittest.TestCase):
    def test_receipt_reproduces_and_validates_end_to_end(self):
        first = run_episode()
        second = run_episode()
        self.assertEqual(first, second)
        validate_receipt(first)
        self.assertEqual(first["assurance_level"], "INTERNAL_PASS")
        self.assertEqual(first["claim_ceiling"], "LocalDevelopmentSyntheticLearningEpisodeReceiptV1")
        self.assertTrue(first["cohorts"]["pairwise_disjoint"])
        self.assertEqual(len(first["selection"]["trace"]), 8)
        self.assertEqual(len(first["updates"]), 8)
        self.assertTrue(all(row["mechanism"]["no_op_identity"] for row in first["updates"]))
        self.assertTrue(all(row["mechanism"]["intervention_reached"] for row in first["updates"]))

    def test_validator_rejects_resealed_metric_tampering(self):
        receipt = run_episode()
        altered = copy.deepcopy(receipt)
        altered["assessment"]["selector_mse"] += 0.25
        altered["receipt_digest"] = digest(
            {key: value for key, value in altered.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(ValidationError, "assessment"):
            validate_receipt(altered)

    def test_assessment_requires_two_valid_locks_and_is_one_use(self):
        receipt = run_episode()
        vault = SealedAssessment()
        with self.assertRaisesRegex(ValueError, "requires selector and control"):
            vault.evaluate({})
        self.assertEqual(vault.phase, "sealed")
        result = vault.evaluate(receipt["prediction_locks"])
        self.assertEqual(vault.phase, "opened")
        self.assertEqual(result["roster_digest"], receipt["assessment"]["roster_digest"])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            vault.evaluate(receipt["prediction_locks"])


if __name__ == "__main__":
    unittest.main()
