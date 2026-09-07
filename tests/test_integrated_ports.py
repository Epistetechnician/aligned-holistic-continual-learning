"""Tests for the integrated upstream state slice."""

# State slice: aligned-holistic-continual-learning-integrated-stack-v2.

import json
import math
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ahcl.adapters import (
    ACDCDiscoveryAdapter,
    CircuitTracerAdapter,
    NNsightActivationAdapter,
    NeuronpediaClient,
    PyveneInterventionAdapter,
    SAELensFeatureAdapter,
    TransformerLensActivationAdapter,
)
from ahcl.aggregate import AggregateRow, reproduce_metrics, reproduce_metrics_from_predictions, reproduce_shuffled_r2
from ahcl.checks import (
    CheckError,
    assert_intervention_reach,
    assert_noop_identity,
    normalized_reconstruction_mse,
)
from ahcl.execution import ExecutionClosed, require_execution_open
from ahcl.custody import CustodyError, build_manifest, validate_manifest
from ahcl.controllers import (
    BehavioralGuardController,
    ControlError,
    FixedUpdateController,
    MonitorControlledUpdatePolicy,
    RetentionGuardController,
)
from ahcl.continual import StreamBatch
from ahcl.lifecycle import LifecycleError, LockedTrial, TrialRow
from ahcl.monitor import RidgeMonitor, metrics, shuffled_metrics
from ahcl.preflight import build_review_packet, validate_dependency_lanes, validate_review_packet
from ahcl.qualification import QualificationThresholds, qualify_path
from ahcl.review import ReviewError, ReviewReceipt, packet_digest, require_independent_accept
from ahcl.stats import StatisticalPlan, achieved_power, holm_adjust, power_check, validate_locked_study
from ahcl.synthetic import AXES, observations


class IntegratedPortTests(unittest.TestCase):
    def test_all_adapters_import_without_loading_optional_packages(self):
        self.assertTrue(all(cls is not None for cls in (
            ACDCDiscoveryAdapter,
            CircuitTracerAdapter,
            NNsightActivationAdapter,
            PyveneInterventionAdapter,
            SAELensFeatureAdapter,
            TransformerLensActivationAdapter,
        )))

    def test_model_and_network_adapters_are_fail_closed(self):
        with self.assertRaises(RuntimeError):
            TransformerLensActivationAdapter.boot()
        with self.assertRaises(RuntimeError):
            SAELensFeatureAdapter.from_pretrained()
        with self.assertRaises(RuntimeError):
            NNsightActivationAdapter.load()
        with self.assertRaises(RuntimeError):
            CircuitTracerAdapter.from_pretrained(transcoder_set="declared-later")
        with self.assertRaises(RuntimeError):
            PyveneInterventionAdapter.from_model(object(), [])
        with self.assertRaises(RuntimeError):
            ACDCDiscoveryAdapter.prepare(object(), object(), object(), threshold=0.1, metric=lambda value: value)
        with self.assertRaises(RuntimeError):
            NeuronpediaClient().post_json("/closed", {})

    def test_submodule_commits_match_provenance_lock(self):
        lock = json.loads(Path("third_party/UPSTREAMS.lock.json").read_text(encoding="utf-8"))
        for item in lock["submodules"]:
            actual = subprocess.check_output(["git", "-C", item["path"], "rev-parse", "HEAD"], text=True).strip()
            self.assertEqual(actual, item["commit"])
            for license_file in item["license_files"]:
                self.assertTrue((Path(item["path"]) / license_file).exists())

    def test_dependency_lanes_match_runtime_lock(self):
        config = json.loads(Path("configs/integrated-stack-v2.json").read_text(encoding="utf-8"))
        digest = validate_dependency_lanes(Path("configs/dependency-lanes.json"), config)
        self.assertEqual(len(digest), 64)

    def test_custody_manifest_hashes_regular_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            item = root / "item.bin"
            item.write_bytes(b"custodied")
            manifest = build_manifest(root, [item], state_slice="aligned-holistic-continual-learning-integrated-stack-v2")
            validate_manifest(manifest, expected_root=root)
            item.write_bytes(b"tampered")
            with self.assertRaises(CustodyError):
                validate_manifest(manifest, expected_root=root)

    def test_review_packet_digest_is_bound(self):
        packet = build_review_packet(
            config={"state_slice": "aligned-holistic-continual-learning-integrated-stack-v2"},
            upstream_lock={"commit": "source"},
            corpus_manifest={"manifest_digest": "corpus"},
            source_digest="source-tree",
        )
        validate_review_packet(packet)
        packet["source_digest"] = "changed"
        with self.assertRaises(ValueError):
            validate_review_packet(packet)

    def test_numerical_checks_enforce_noop_and_reach(self):
        self.assertEqual(assert_noop_identity((1.0, 2.0), (1.0, 2.0)), 0.0)
        self.assertEqual(normalized_reconstruction_mse((1.0, 2.0), (1.0, 2.0)), 0.0)
        self.assertGreater(assert_intervention_reach((0.0,), (0.5,)), 0.0)
        with self.assertRaises(CheckError):
            assert_intervention_reach((0.0,), (0.0,))
        with self.assertRaises(CheckError):
            normalized_reconstruction_mse((math.nan,), (0.0,))

    def test_combined_path_qualification_runs_all_checks(self):
        evidence = qualify_path(
            native_output=(1.0, 2.0),
            instrumented_output=(1.0, 2.0),
            repeated_output=(1.0, 2.0),
            no_op_output=(1.0, 2.0),
            intervened_output=(1.0, 2.5),
            base_output=(1.0, 2.0),
            activation=(1.0, 2.0),
            reconstructed_activation=(1.0, 2.01),
            clean_output=(1.0, 2.0),
            scrubbed_output=(1.0, 2.2),
            thresholds=QualificationThresholds(reconstruction_nmse_max=0.01),
        )
        self.assertEqual(evidence.parity_delta, 0.0)
        self.assertGreater(evidence.intervention_delta, 0.0)
        with self.assertRaises(ValueError):
            qualify_path(
                native_output=(1.0,),
                instrumented_output=(1.0,),
                repeated_output=(1.0,),
                no_op_output=(1.0,),
                intervened_output=(1.0,),
                base_output=(1.0,),
                activation=(1.0,),
                reconstructed_activation=(2.0,),
                clean_output=(1.0,),
                scrubbed_output=(1.1,),
            )

    def test_holm_and_power_contract(self):
        self.assertEqual(holm_adjust((0.01, 0.04)), (0.02, 0.04))
        plan = StatisticalPlan()
        power_check(plan)
        self.assertGreaterEqual(achieved_power(plan), plan.power)

    def test_machine_readable_study_lock_is_frozen(self):
        config = json.loads(Path("configs/integrated-stack-v2.json").read_text(encoding="utf-8"))
        plan = validate_locked_study(config)
        self.assertEqual(plan.as_config()["family_counts"], {"fit": 48, "tune": 48, "assessment": 48})
        altered = dict(config)
        altered["controls"] = list(reversed(altered["controls"]))
        with self.assertRaises(ValueError):
            validate_locked_study(altered)

    def test_monitor_controller_gates_updates_and_behavioral_comparator(self):
        class CountingPolicy:
            def __init__(self):
                self.calls = 0

            def update(self, _batch):
                self.calls += 1
                return 1.25

        batch = StreamBatch("b1", (), (), {})
        base = CountingPolicy()
        controller = RetentionGuardController(retention_harm_ceiling=0.2)
        policy = MonitorControlledUpdatePolicy(
            base,
            feature_extractor=lambda _batch: (1.0, 2.0, 3.0),
            predictor=lambda _features: (0.0, 0.3, 0.0, 0.0),
            controller=controller,
        )
        self.assertEqual(policy.update(batch), 0.0)
        self.assertEqual(base.calls, 0)
        self.assertEqual(policy.last_decision.reason, "predicted_retention_harm")

        policy.predictor = lambda _features: (0.2, 0.1, 0.0, 0.0)
        self.assertEqual(policy.update(batch), 1.25)
        self.assertEqual(base.calls, 1)
        self.assertEqual(policy.control_summary()["applied_updates"], 1)
        self.assertEqual(FixedUpdateController().decide((0.0,)).action, "apply")

        always_skip = MonitorControlledUpdatePolicy(
            CountingPolicy(),
            feature_extractor=lambda _batch: (1.0,),
            predictor=lambda _features: (0.0, 1.0),
            controller=RetentionGuardController(retention_harm_ceiling=0.2),
        )
        always_skip.update(batch)
        with self.assertRaises(ControlError):
            always_skip.control_summary()

        behavior = BehavioralGuardController(retention_harm_ceiling=0.2)
        self.assertEqual(behavior.decide_from_metrics({"retention_harm": 0.3}).action, "skip")
        with self.assertRaises(ControlError):
            behavior.decide_from_metrics({"retention_harm": float("nan")})

    def test_independent_aggregates_match_locked_monitor(self):
        rows = observations("assessment", "fixed_update")
        monitor = RidgeMonitor.fit(observations("fit", "fixed_update"), ridge=0.1)
        independent_rows = tuple(
            AggregateRow(str(row.family), row.split, 0, row.features, row.outcomes) for row in rows
        )
        reproduced = reproduce_metrics(independent_rows, monitor.predict)
        reproduced_from_packet = reproduce_metrics_from_predictions(
            independent_rows,
            tuple(monitor.predict(row.features) for row in independent_rows),
        )
        expected = metrics(rows, monitor)
        for index, axis in enumerate(AXES):
            self.assertAlmostEqual(reproduced[str(index)]["r2"], expected[axis]["r2"])
            self.assertAlmostEqual(reproduced[str(index)]["sign_agreement"], expected[axis]["sign_agreement"])
            self.assertEqual(reproduced_from_packet[str(index)], reproduced[str(index)])
        shuffled = reproduce_shuffled_r2(independent_rows, monitor.predict)
        expected_shuffled = shuffled_metrics(rows, monitor)
        for index, axis in enumerate(AXES):
            self.assertAlmostEqual(shuffled[str(index)], expected_shuffled[axis])

    def test_locked_trial_enforces_phase_order_and_disjoint_cohorts(self):
        plan = StatisticalPlan(
            fit_families=2,
            tune_families=2,
            assessment_families=2,
            repeats=2,
            power=0.01,
        )
        config = {"state_slice": "aligned-holistic-continual-learning-integrated-stack-v2"}
        trial = LockedTrial(config, plan)

        def rows(split, offset):
            return tuple(
                TrialRow(f"f{offset + family}", split, repeat, (float(family),), (float(repeat),))
                for family in range(2)
                for repeat in range(2)
            )

        with self.assertRaises(LifecycleError):
            trial.tune(rows("tune", 2), lambda _predictor, _rows: {"score": 1.0})
        trial.fit(rows("fit", 0), lambda fit_rows: ("predictor", {"kind": "test"}))
        lock = trial.tune(rows("tune", 2), lambda _predictor, _rows: {"score": 1.0})
        self.assertEqual(trial.phase, "locked")
        self.assertEqual(lock.tune_summary, {"score": 1.0})
        self.assertEqual(trial.assess(rows("assessment", 4), lambda _predictor, _rows: {"score": 2.0}), {"score": 2.0})
        with self.assertRaises(LifecycleError):
            trial.assess(rows("assessment", 0), lambda _predictor, _rows: {"score": 2.0})

    def test_operator_cannot_self_accept(self):
        packet = {"state_slice": "aligned-holistic-continual-learning-integrated-stack-v2"}
        receipt = ReviewReceipt(packet_digest(packet), "operator", "ACCEPT", "signed")
        with self.assertRaises(ReviewError):
            require_independent_accept(receipt, expected_packet_digest=packet_digest(packet), operator_id="operator")

    def test_review_requires_independent_signature_verifier(self):
        packet = {"state_slice": "aligned-holistic-continual-learning-integrated-stack-v2"}
        receipt = ReviewReceipt(packet_digest(packet), "reviewer", "ACCEPT", "signed", public_key_id="key-1")
        with self.assertRaises(ReviewError):
            require_independent_accept(receipt, expected_packet_digest=packet_digest(packet), operator_id="operator")

    def test_execution_stays_closed_without_review(self):
        config = {"execution": {"enabled": False}}
        with self.assertRaises(ExecutionClosed):
            require_execution_open(config, None, packet_digest="packet", operator_id="operator")


if __name__ == "__main__":
    unittest.main()
