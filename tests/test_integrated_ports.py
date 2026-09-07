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
from ahcl.checks import (
    CheckError,
    assert_intervention_reach,
    assert_noop_identity,
    normalized_reconstruction_mse,
)
from ahcl.execution import ExecutionClosed, require_execution_open
from ahcl.custody import CustodyError, build_manifest, validate_manifest
from ahcl.preflight import build_review_packet, validate_review_packet
from ahcl.review import ReviewError, ReviewReceipt, packet_digest, require_independent_accept
from ahcl.stats import StatisticalPlan, holm_adjust, power_check


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

    def test_holm_and_power_contract(self):
        self.assertEqual(holm_adjust((0.01, 0.04)), (0.02, 0.04))
        power_check(StatisticalPlan())

    def test_operator_cannot_self_accept(self):
        packet = {"state_slice": "aligned-holistic-continual-learning-integrated-stack-v2"}
        receipt = ReviewReceipt(packet_digest(packet), "operator", "ACCEPT", "signed")
        with self.assertRaises(ReviewError):
            require_independent_accept(receipt, expected_packet_digest=packet_digest(packet), operator_id="operator")

    def test_execution_stays_closed_without_review(self):
        config = {"execution": {"enabled": False}}
        with self.assertRaises(ExecutionClosed):
            require_execution_open(config, None, packet_digest="packet", operator_id="operator")


if __name__ == "__main__":
    unittest.main()
