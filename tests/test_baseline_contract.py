"""State slice: aligned-holistic-continual-learning-integrated-stack-v2."""

import unittest
from ahcl.baseline_contract import (
    BASELINES, BudgetExceeded, Cohort, Cost, ResourceCaps, ResourceLedger,
    learning_floor_passes, validate_comparison, validate_splits,
)


class BaselineContractTests(unittest.TestCase):
    def test_every_overhead_consumes_the_same_budget(self):
        ledger = ResourceLedger(ResourceCaps(80, 100, 80))
        for category in ("setup", "update", "replay", "fisher", "evaluation", "monitor", "candidate", "rollback"):
            ledger.charge(Cost(category, 10, 90, 10))
        self.assertEqual(ledger.compute_ops, 80)
        self.assertEqual(ledger.peak_memory_bytes, 90)
        with self.assertRaises(BudgetExceeded):
            ledger.preflight(Cost("update", 1, 90, 0))
        with self.assertRaises(BudgetExceeded):
            ledger.charge(Cost("update", 0, 0, 0))

    def test_memory_and_elapsed_are_independent_hard_limits(self):
        for cost in (Cost("candidate", 0, 101, 0), Cost("evaluation", 0, 0, 101)):
            ledger = ResourceLedger(ResourceCaps(100, 100, 100))
            with self.assertRaises(BudgetExceeded):
                ledger.preflight(cost)
            self.assertEqual(ledger.compute_ops, 0)

    def test_negative_or_unknown_costs_cannot_credit_budget(self):
        for args in (("update", -1, 0, 0), ("hidden", 1, 0, 0), ("update", True, 0, 0)):
            with self.assertRaises(ValueError):
                Cost(*args)

    def test_equal_caps_and_complete_roster(self):
        caps = {arm: ResourceCaps(100, 100, 100) for arm in BASELINES}
        validate_comparison(caps)
        caps["causal_monitor"] = ResourceCaps(101, 100, 100)
        with self.assertRaises(ValueError):
            validate_comparison(caps)

    def test_document_and_author_leakage_despite_unique_families(self):
        splits = {name: Cohort(frozenset({name}), frozenset({name}), frozenset({name}))
                  for name in ("fit", "tune", "assessment")}
        validate_splits(splits)
        for documents, authors in (({"fit"}, {"assessment"}), ({"assessment"}, {"fit"})):
            splits["assessment"] = Cohort(frozenset({"assessment"}), frozenset(documents), frozenset(authors))
            with self.assertRaises(ValueError):
                validate_splits(splits)

    def test_reject_all_incomplete_and_nan_cannot_pass(self):
        ledger = ResourceLedger(ResourceCaps(100, 100, 100))
        args = dict(gain_lower_bound=.2, minimum_gain=.1, accepted_updates=1,
                    ledger=ledger, completed_stream=True)
        self.assertTrue(learning_floor_passes(**args))
        for change in ({"accepted_updates": 0}, {"completed_stream": False},
                       {"gain_lower_bound": float("nan")}, {"gain_lower_bound": .09}):
            self.assertFalse(learning_floor_passes(**(args | change)))


if __name__ == "__main__":
    unittest.main()
