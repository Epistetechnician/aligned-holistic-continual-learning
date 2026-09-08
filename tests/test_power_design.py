"""Tests for the conservative document-level power planner.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

import math
import unittest

from ahcl.power_design import PowerDesign, two_sided_power
from ahcl.stats import StatisticalPlan, familywise_primary_claim


class PowerDesignTests(unittest.TestCase):
    def test_single_endpoint_is_hand_calculable_and_monotone(self):
        plan = PowerDesign(effect_size=1.0, endpoints=1, repeats=1, max_clusters=100)
        self.assertAlmostEqual(plan.per_contrast_alpha, 0.05)
        self.assertEqual(plan.marginal_required_clusters, 8)
        self.assertGreater(plan.marginal_power(20), plan.marginal_power(10))

    def test_multiplicity_increases_required_documents(self):
        one = PowerDesign(effect_size=0.5, endpoints=1, max_clusters=2_000)
        many = PowerDesign(effect_size=0.5, endpoints=5, max_clusters=2_000)
        self.assertAlmostEqual(many.per_contrast_alpha, 0.01)
        self.assertGreater(many.marginal_required_clusters, one.marginal_required_clusters)

    def test_repeats_do_not_inflate_independent_document_count(self):
        one = PowerDesign(effect_size=0.8, endpoints=1, repeats=1, max_clusters=1_000)
        repeated = PowerDesign(effect_size=0.8, endpoints=1, repeats=3, max_clusters=1_000)
        self.assertEqual(repeated.marginal_required_clusters, one.marginal_required_clusters)
        self.assertAlmostEqual(
            two_sided_power(
                clusters=one.marginal_required_clusters,
                effect_size=0.8,
                alpha=0.05,
                repeats=1,
            ),
            one.marginal_power(one.marginal_required_clusters),
        )

    def test_joint_target_is_distinct_and_uses_union_bound(self):
        plan = PowerDesign(effect_size=1.0, endpoints=2, joint_target=0.80, max_clusters=1_000)
        clusters = plan.joint_required_clusters
        self.assertIsNotNone(clusters)
        assert clusters is not None
        self.assertGreaterEqual(plan.joint_lower_bound(clusters), 0.80)
        if clusters > 1:
            self.assertLess(plan.joint_lower_bound(clusters - 1), 0.80)
        self.assertAlmostEqual(
            plan.joint_lower_bound(clusters),
            max(0.0, 1 - 2 * (1 - plan.marginal_power(clusters))),
        )

    def test_familywise_requirement_expands_to_families_per_cohort(self):
        plan = PowerDesign(effect_size=0.5, endpoints=4, joint_target=0.80, max_clusters=1_000)
        self.assertEqual(plan.joint_required_clusters, 69)
        self.assertEqual(plan.required_families_per_cohort(families_per_document=8), 552)

    def test_current_six_document_panel_is_not_declared_powered(self):
        plan = PowerDesign(effect_size=0.5, endpoints=4, repeats=3, joint_target=0.80)
        self.assertGreater(plan.joint_required_clusters, 6)
        self.assertLess(plan.joint_lower_bound(6), 0.80)

    def test_current_panel_returns_no_candidate_and_unvalidated(self):
        outcome = familywise_primary_claim(StatisticalPlan())
        self.assertEqual(outcome.status, "NoCandidate")
        self.assertEqual(outcome.validation, "unvalidated")
        self.assertEqual(outcome.available_documents, 6)
        self.assertEqual(outcome.required_documents, 69)
        self.assertEqual(outcome.required_families_per_cohort, 552)
        self.assertLess(outcome.power_lower_bound, outcome.target_power)

    def test_familywise_cohort_size_is_monotone_and_repeats_invariant(self):
        smaller = StatisticalPlan(
            fit_families=551, tune_families=551, assessment_families=551
        )
        required = StatisticalPlan(
            fit_families=552, tune_families=552, assessment_families=552
        )
        repeated = StatisticalPlan(
            fit_families=552,
            tune_families=552,
            assessment_families=552,
            repeats=7,
        )
        smaller_outcome = familywise_primary_claim(smaller)
        required_outcome = familywise_primary_claim(required)
        repeated_outcome = familywise_primary_claim(repeated)
        self.assertEqual(smaller_outcome.status, "NoCandidate")
        self.assertEqual(required_outcome.status, "Candidate")
        self.assertEqual(required_outcome.power_lower_bound, repeated_outcome.power_lower_bound)
        self.assertEqual(required_outcome.required_documents, repeated_outcome.required_documents)

    def test_bonferroni_is_safe_for_the_same_familywise_calculation(self):
        outcome = familywise_primary_claim(
            StatisticalPlan(
                fit_families=552,
                tune_families=552,
                assessment_families=552,
                multiplicity="Bonferroni",
            )
        )
        self.assertEqual(outcome.status, "Candidate")
        self.assertAlmostEqual(outcome.per_contrast_alpha, 0.0125)

    def test_unmet_document_assumption_is_no_candidate(self):
        outcome = familywise_primary_claim(StatisticalPlan(independent_unit="family"))
        self.assertEqual(outcome.status, "NoCandidate")
        self.assertEqual(outcome.validation, "unvalidated")
        self.assertIn("independent_unit", outcome.reason)

    def test_no_complete_document_is_no_candidate(self):
        outcome = familywise_primary_claim(
            StatisticalPlan(fit_families=7, tune_families=7, assessment_families=7)
        )
        self.assertEqual(outcome.status, "NoCandidate")
        self.assertEqual(outcome.available_documents, 0)
        self.assertEqual(outcome.power_lower_bound, 0.0)

    def test_nonfinite_inputs_are_rejected(self):
        for field in ("effect_size", "alpha", "target_power"):
            values = {"effect_size": 0.5}
            values[field] = math.nan
            with self.subTest(field=field), self.assertRaises(ValueError):
                PowerDesign(**values)
        with self.assertRaises(ValueError):
            familywise_primary_claim(StatisticalPlan(alpha=math.nan))

    def test_counts_must_be_positive_integers(self):
        for field in ("endpoints", "repeats", "max_clusters"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                PowerDesign(effect_size=0.5, **{field: 0})

    def test_familywise_conversion_rejects_malformed_family_count(self):
        plan = PowerDesign(effect_size=0.5, endpoints=4, joint_target=0.80)
        with self.assertRaises(ValueError):
            plan.required_families_per_cohort(families_per_document=0)


if __name__ == "__main__":
    unittest.main()
