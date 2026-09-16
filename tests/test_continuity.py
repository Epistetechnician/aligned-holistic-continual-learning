"""Mechanistic continuity contract tests.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

import unittest

from ahcl.continuity import (
    CheckpointMeasurement,
    ContinuityError,
    ContinuityPolicy,
    LineageEdge,
    LineageTransition,
    build_continuity_report,
)


def checkpoint(number: int, *, circuits: dict[str, float] | None = None, behavior=(1.0, 0.5)) -> CheckpointMeasurement:
    return CheckpointMeasurement(
        checkpoint_id=f"c{number}",
        step=number,
        state_digest=f"{number + 1:064x}",
        representation=(1.0, 0.2 + number * 0.01, 0.5),
        circuit_effects=circuits or {"safe": 1.0, "task": 0.8},
        intervention_effects={"refusal": 1.0, "task": 0.8},
        behavior=behavior,
    )


class ContinuityTests(unittest.TestCase):
    def test_end_to_end_preserved_lineage_passes_and_is_digest_bound(self):
        before, after = checkpoint(0), checkpoint(1)
        report = build_continuity_report(
            (before, after),
            (LineageTransition("c0", "c1", (LineageEdge("safe", "safe", 1.0), LineageEdge("task", "task", 1.0))),),
        )
        self.assertTrue(report["all_invariants_pass"])
        self.assertFalse(report["mechanistic_drift_detected"])
        self.assertEqual(len(report["report_digest"]), 64)
        self.assertEqual(report["transitions"][0]["failed_invariants"], ())
        changed_identity = CheckpointMeasurement(
            after.checkpoint_id,
            after.step,
            "f" * 64,
            after.representation,
            after.circuit_effects,
            after.intervention_effects,
            after.behavior,
        )
        changed_report = build_continuity_report(
            (before, changed_identity),
            (LineageTransition("c0", "c1", (LineageEdge("safe", "safe", 1.0), LineageEdge("task", "task", 1.0))),),
        )
        self.assertNotEqual(report["report_digest"], changed_report["report_digest"])

    def test_split_and_disappearance_are_explicitly_scored(self):
        split = build_continuity_report(
            (checkpoint(0, circuits={"safe": 1.0}), checkpoint(1, circuits={"safe_a": 0.5, "safe_b": 0.5})),
            (LineageTransition("c0", "c1", (LineageEdge("safe", "safe_a", 0.5), LineageEdge("safe", "safe_b", 0.5))),),
            policy=ContinuityPolicy(circuit_min=0.8),
        )
        self.assertEqual(split["transitions"][0]["source_preservation"], 1.0)
        self.assertEqual(split["transitions"][0]["target_coverage"], 0.5)
        self.assertFalse(split["all_invariants_pass"])

        disappeared = build_continuity_report(
            (checkpoint(0), checkpoint(1, circuits={"new": 1.0})),
            (LineageTransition("c0", "c1", ()),),
        )
        self.assertEqual(disappeared["transitions"][0]["circuit_continuity"], 0.0)

    def test_behaviorally_equivalent_mechanistic_drift_is_detected(self):
        report = build_continuity_report(
            (checkpoint(0, circuits={"old": 1.0}), checkpoint(1, circuits={"new": 1.0}, behavior=(1.0, 0.5))),
            (LineageTransition("c0", "c1", ()),),
            policy=ContinuityPolicy(behavior_min=0.95),
        )
        row = report["transitions"][0]
        self.assertTrue(row["behaviorally_equivalent_mechanistic_drift"])
        self.assertTrue(report["mechanistic_drift_detected"])
        self.assertIn("circuit", row["failed_invariants"])

    def test_lineage_is_not_allowed_to_rematch_or_overclaim(self):
        with self.assertRaisesRegex(ContinuityError, "order exactly"):
            build_continuity_report(
                (checkpoint(0), checkpoint(1)),
                (LineageTransition("c1", "c0", ()),),
            )
        with self.assertRaisesRegex(ContinuityError, "absent"):
            build_continuity_report(
                (checkpoint(0), checkpoint(1)),
                (LineageTransition("c0", "c1", (LineageEdge("missing", "safe", 1.0),)),),
            )
        self.assertIn("no mechanistic understanding claim", build_continuity_report(
            (checkpoint(0), checkpoint(1)),
            (LineageTransition("c0", "c1", (LineageEdge("safe", "safe", 1.0), LineageEdge("task", "task", 1.0))),),
        )["nonclaims"])

    def test_trajectory_and_identity_inputs_fail_closed(self):
        with self.assertRaisesRegex(ContinuityError, "complete and ordered"):
            build_continuity_report(
                (checkpoint(0), checkpoint(2)),
                (LineageTransition("c0", "c2", ()),),
            )
        with self.assertRaisesRegex(ContinuityError, "lowercase"):
            CheckpointMeasurement("bad", 0, "A" * 64, (1,), {"x": 1}, {"p": 1}, (1,))
        with self.assertRaisesRegex(ContinuityError, "mass"):
            LineageTransition("c0", "c1", (LineageEdge("safe", "safe", 0.8), LineageEdge("safe", "task", 0.5)))


if __name__ == "__main__":
    unittest.main()
