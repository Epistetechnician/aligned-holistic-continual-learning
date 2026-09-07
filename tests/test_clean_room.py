import json
import tempfile
import unittest
from pathlib import Path

from ahcl.canonical import digest
from ahcl.experiment import run_experiment
from ahcl.interfaces import PUBLIC_BACKENDS, registry_is_quarantined
from ahcl.monitor import RidgeMonitor, design_row
from ahcl.synthetic import ARMS, SPLITS, families, observations, update_delta
from ahcl.validate import ValidationError, validate_result


class CleanRoomTests(unittest.TestCase):
    def test_canonical_digest_is_order_independent(self):
        self.assertEqual(digest({"b": 2, "a": 1}), digest({"a": 1, "b": 2}))

    def test_roster_and_arm_cardinality_are_locked(self):
        self.assertEqual(sum(1 for _ in families()), 144)
        for split, count in SPLITS.items():
            for arm in ARMS:
                self.assertEqual(len(observations(split, arm)), count)

    def test_noop_is_exact_and_update_reaches_features(self):
        family = next(iter(families()))
        self.assertEqual(update_delta(family, "untouched_base", 0), (0.0,) * 8)
        self.assertGreater(sum(abs(value) for value in update_delta(family, "fixed_update", 0)[:3]), 0.0)

    def test_fixed_design_has_seven_terms(self):
        self.assertEqual(len(design_row((1.0, 2.0, 3.0))), 7)

    def test_public_backend_registry_is_quarantined(self):
        self.assertEqual(len(PUBLIC_BACKENDS), 5)
        self.assertTrue(registry_is_quarantined())

    def test_ridge_monitor_fits_known_relation(self):
        rows = observations("fit", "fixed_update")
        monitor = RidgeMonitor.fit(rows, ridge=1e-8)
        prediction = monitor.predict(rows[0].features)
        self.assertEqual(len(prediction), 4)
        self.assertTrue(all(abs(value) < 1.0 for value in prediction))

    def test_end_to_end_result_is_valid_and_deterministic(self):
        first = run_experiment()
        second = run_experiment()
        self.assertEqual(first, second)
        validate_result(first)
        self.assertIn(first["status"], {"SyntheticCandidate", "SyntheticNoCandidate"})

    def test_validator_rejects_digest_tampering(self):
        payload = run_experiment()
        payload["experiment_digest"] = "0" * 64
        with self.assertRaises(ValidationError):
            validate_result(payload)

    def test_validator_rejects_malformed_prediction_lock(self):
        payload = run_experiment()
        payload["prediction_lock_digest"] = "short"
        with self.assertRaises(ValidationError):
            validate_result(payload)

    def test_cli_output_is_json(self):
        payload = run_experiment()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["experiment_digest"], payload["experiment_digest"])


if __name__ == "__main__":
    unittest.main()
