"""Tests for the integrated upstream state slice."""

# State slice: aligned-holistic-continual-learning-integrated-stack-v2.

import json
import hashlib
import math
import subprocess
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from ahcl.adapters import (
    ACDCDiscoveryAdapter,
    CircuitTracerAdapter,
    NNsightActivationAdapter,
    NeuronpediaClient,
    PyveneInterventionAdapter,
    SAELensFeatureAdapter,
    TransformerLensActivationAdapter,
)
from ahcl.adapters.common import (
    validate_imported_module,
    validate_local_custody_file,
    validate_local_custody_tree,
)
from ahcl.aggregate import AggregateError, AggregateRow, reproduce_metrics, reproduce_metrics_from_predictions, reproduce_shuffled_r2
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
from ahcl.preflight import (
    build_review_packet,
    validate_corpus_custody,
    validate_corpus_manifest,
    validate_dependency_lanes,
    validate_lane_sources,
    validate_review_packet,
)
from ahcl.qualification import QualificationThresholds, qualify_path
from ahcl.review import (
    Ed25519Verifier,
    ReviewError,
    ReviewReceipt,
    canonical_review_message,
    packet_digest,
    require_independent_accept,
)
from ahcl.stats import (
    StatisticalPlan,
    achieved_power,
    document_power_lower_bound,
    holm_adjust,
    power_check,
    validate_locked_study,
)
from ahcl.synthetic import AXES, observations
from ahcl.runtime_attestation import (
    RuntimeAttestation,
    RuntimeAttestationError,
    RuntimeLaneAttestation,
    validate_runtime_attestation,
)


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

    def test_local_adapter_inputs_are_hash_and_symlink_bound(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "weights.safetensors"
            item.write_bytes(b"local-custody")
            checksum = hashlib.sha256(item.read_bytes()).hexdigest()
            self.assertEqual(validate_local_custody_file(item, checksum, "weights"), item.resolve())
            link = root / "link.safetensors"
            link.symlink_to(item)
            with self.assertRaises(RuntimeError):
                validate_local_custody_file(link, checksum, "weights")

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
        validate_lane_sources(Path("configs/dependency-lanes.json"), Path("third_party/UPSTREAMS.lock.json"))

    def test_corpus_manifest_is_digest_bound_and_disjoint(self):
        manifest = json.loads(Path("configs/corpus-manifest.v2.json").read_text(encoding="utf-8"))
        config = json.loads(Path("configs/integrated-stack-v2.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_corpus_manifest(manifest, config=config), manifest["manifest_digest"])
        custody = {
            "files": [
                {"path": f"corpus/{document['id']}.txt", "sha256": document["sha256"], "bytes": 1}
                for document in manifest["documents"]
            ]
        }
        validate_corpus_custody(manifest, custody)
        altered = dict(manifest)
        altered["documents"] = list(manifest["documents"])
        altered["documents"][1] = dict(altered["documents"][1], author=altered["documents"][0]["author"])
        with self.assertRaises(ValueError):
            validate_corpus_manifest(altered, config=config)
        altered = dict(manifest)
        altered["documents"] = list(manifest["documents"])
        altered["documents"][1] = dict(
            altered["documents"][1],
            family_ids=list(altered["documents"][0]["family_ids"]),
        )
        with self.assertRaises(ValueError):
            validate_corpus_manifest(altered, config=config)

    def test_custody_manifest_hashes_regular_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            item = root / "item.bin"
            item.write_bytes(b"custodied")
            manifest = build_manifest(root, [item], state_slice="aligned-holistic-continual-learning-integrated-stack-v2")
            validate_manifest(manifest, expected_root=root)
            duplicate = dict(manifest)
            duplicate["files"] = list(manifest["files"]) + [dict(manifest["files"][0])]
            duplicate["manifest_digest"] = packet_digest({key: value for key, value in duplicate.items() if key != "manifest_digest"})
            with self.assertRaises(CustodyError):
                validate_manifest(duplicate, expected_root=root)
            item.write_bytes(b"tampered")
            with self.assertRaises(CustodyError):
                validate_manifest(manifest, expected_root=root)

    def test_adapter_loader_identity_checks_are_local_and_exact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.safetensors"
            weights.write_bytes(b"locked bytes")
            checksum = hashlib.sha256(weights.read_bytes()).hexdigest()
            self.assertEqual(validate_local_custody_file(weights, checksum, "weights"), weights.resolve())
            with self.assertRaises(RuntimeError):
                validate_local_custody_file(weights, "0" * 64, "weights")
            with self.assertRaises(RuntimeError):
                validate_imported_module(json, root, "test module")

    def test_imported_module_requires_exact_source_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            module_file = root / "nnsight" / "__init__.py"
            module_file.parent.mkdir()
            module_file.write_bytes(b"pinned source")
            expected = {
                "nnsight/__init__.py": hashlib.sha256(module_file.read_bytes()).hexdigest(),
            }
            module = SimpleNamespace(__file__=str(module_file))
            self.assertEqual(
                validate_imported_module(module, root, "NNsight", expected_files=expected),
                module_file.resolve(),
            )
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "__init__.cpython-314.pyc").write_bytes(b"transient bytecode")
            self.assertEqual(
                validate_imported_module(module, root, "NNsight", expected_files=expected),
                module_file.resolve(),
            )
            module_file.write_bytes(b"changed source")
            with self.assertRaisesRegex(RuntimeError, "digest"):
                validate_imported_module(module, root, "NNsight", expected_files=expected)
            module_file.write_bytes(b"pinned source")
            (root / "extra.py").write_bytes(b"roster drift")
            with self.assertRaisesRegex(RuntimeError, "roster"):
                validate_imported_module(module, root, "NNsight", expected_files=expected)
            (root / "extra.py").unlink()
            with self.assertRaisesRegex(RuntimeError, "invalid path"):
                validate_imported_module(
                    module,
                    root,
                    "NNsight",
                    expected_files={"../outside.py": expected["nnsight/__init__.py"]},
                )
            alias = root / "alias.py"
            alias.symlink_to(module_file)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                validate_imported_module(module, root, "NNsight", expected_files={
                    **expected,
                    "alias.py": expected["nnsight/__init__.py"],
                })

    def test_nnsight_loader_rejects_source_roster_before_model_load(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            module_file = source_root / "nnsight" / "__init__.py"
            module_file.parent.mkdir(parents=True)
            module_file.write_bytes(b"pinned nnsight")
            model_root = root / "model"
            model_root.mkdir()
            weights = model_root / "model.safetensors"
            weights.write_bytes(b"weights")
            source_files = {
                "nnsight/__init__.py": hashlib.sha256(module_file.read_bytes()).hexdigest(),
            }
            fake_nnsight = SimpleNamespace(__file__=str(module_file))
            fake_nnsight.LanguageModel = object()
            model_load = Mock()
            fake_transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=model_load),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_load),
            )
            with patch(
                "ahcl.adapters.common.importlib.import_module",
                side_effect=lambda name: {
                    "nnsight": fake_nnsight,
                    "transformers": fake_transformers,
                    "torch": SimpleNamespace(float32=object()),
                }[name],
            ):
                module_file.write_bytes(b"roster drift")
                with self.assertRaisesRegex(RuntimeError, "digest"):
                    NNsightActivationAdapter.load(
                        revision="locked",
                        model_root=model_root,
                        model_weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                        source_root=source_root,
                        source_files=source_files,
                        enabled=True,
                    )
            model_load.assert_not_called()

    def test_circuit_tracer_rejects_module_path_escape_before_model_load(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            in_tree = source_root / "replacement_model.py"
            in_tree.write_bytes(b"manifest entry")
            outside = root / "outside.py"
            outside.write_bytes(b"outside source")
            model_root = root / "model"
            model_root.mkdir()
            weights = model_root / "model.safetensors"
            weights.write_bytes(b"weights")
            transcoder_root = root / "transcoder"
            transcoder_root.mkdir()
            (transcoder_root / "config.yaml").write_text(
                "model_kind: transcoder_set\ntranscoders: [layer.safetensors]\n",
                encoding="utf-8",
            )
            (transcoder_root / "layer.safetensors").write_bytes(b"transcoder")
            transcoder_files = {
                name: hashlib.sha256((transcoder_root / name).read_bytes()).hexdigest()
                for name in ("config.yaml", "layer.safetensors")
            }
            fake_replacement = SimpleNamespace(__file__=str(outside))
            model_load = Mock()
            fake_transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=model_load),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_load),
            )
            with patch(
                "ahcl.adapters.common.importlib.import_module",
                side_effect=lambda name: {
                    "circuit_tracer.replacement_model": fake_replacement,
                    "yaml": SimpleNamespace(safe_load=lambda value: {"model_kind": "transcoder_set", "transcoders": ["layer.safetensors"]}),
                    "torch": SimpleNamespace(float32=object()),
                    "transformers": fake_transformers,
                }[name],
            ):
                with self.assertRaisesRegex(RuntimeError, "outside"):
                    CircuitTracerAdapter.from_pretrained(
                        transcoder_set=str(transcoder_root),
                        revision="locked",
                        model_root=model_root,
                        model_weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                        source_root=source_root,
                        source_files={
                            "replacement_model.py": hashlib.sha256(in_tree.read_bytes()).hexdigest(),
                        },
                        transcoder_root=transcoder_root,
                        transcoder_files=transcoder_files,
                        enabled=True,
                    )
            model_load.assert_not_called()

    def test_circuit_tracer_rejects_submodule_escape_before_model_load(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            replacement_file = source_root / "replacement_model.py"
            replacement_file.write_bytes(b"manifest entry")
            outside = root / "single_layer_transcoder.py"
            outside.write_bytes(b"outside source")
            model_root = root / "model"
            model_root.mkdir()
            weights = model_root / "model.safetensors"
            weights.write_bytes(b"weights")
            transcoder_root = root / "transcoder"
            transcoder_root.mkdir()
            (transcoder_root / "config.yaml").write_text(
                "model_kind: transcoder_set\ntranscoders: [layer.safetensors]\n",
                encoding="utf-8",
            )
            (transcoder_root / "layer.safetensors").write_bytes(b"transcoder")
            transcoder_files = {
                name: hashlib.sha256((transcoder_root / name).read_bytes()).hexdigest()
                for name in ("config.yaml", "layer.safetensors")
            }
            fake_replacement = SimpleNamespace(__file__=str(replacement_file))
            fake_submodule = SimpleNamespace(__file__=str(outside))
            model_load = Mock()
            fake_transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=model_load),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_load),
            )
            with patch(
                "ahcl.adapters.common.importlib.import_module",
                side_effect=lambda name: {
                    "circuit_tracer.replacement_model": fake_replacement,
                    "circuit_tracer.transcoder.single_layer_transcoder": fake_submodule,
                    "yaml": SimpleNamespace(safe_load=lambda value: {"model_kind": "transcoder_set", "transcoders": ["layer.safetensors"]}),
                    "torch": SimpleNamespace(float32=object()),
                    "transformers": fake_transformers,
                }[name],
            ):
                with self.assertRaisesRegex(RuntimeError, "outside"):
                    CircuitTracerAdapter.from_pretrained(
                        transcoder_set=str(transcoder_root),
                        revision="locked",
                        model_root=model_root,
                        model_weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                        source_root=source_root,
                        source_files={
                            "replacement_model.py": hashlib.sha256(replacement_file.read_bytes()).hexdigest(),
                        },
                        transcoder_root=transcoder_root,
                        transcoder_files=transcoder_files,
                        enabled=True,
                    )
            model_load.assert_not_called()

    def test_local_custody_tree_has_an_exact_roster(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.yaml").write_text("model_kind: transcoder_set\n", encoding="utf-8")
            (root / "layer_0.safetensors").write_bytes(b"transcoder")
            expected = {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("config.yaml", "layer_0.safetensors")
            }
            checked = validate_local_custody_tree(root, expected, "transcoder")
            self.assertEqual(tuple(sorted(path.name for path in checked)), ("config.yaml", "layer_0.safetensors"))
            (root / "extra.bin").write_bytes(b"extra")
            with self.assertRaises(RuntimeError):
                validate_local_custody_tree(root, expected, "transcoder")
            (root / "extra.bin").unlink()
            (root / "layer_0.safetensors").write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                validate_local_custody_tree(root, expected, "transcoder")

    def test_runtime_attestation_schema_is_canonical_and_fail_closed(self):
        payload = {
            "schema": "ahcl.runtime.attestation.v1",
            "state_slice": "aligned-holistic-continual-learning-integrated-stack-v2",
            "python": "3.11.11",
            "interpreter": {
                "path": str(Path(sys.executable).resolve()),
                "sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
            },
            "distributions": {
                "torch": {
                    "distribution": "torch",
                    "version": "2.6.0",
                    "import_name": "torch",
                    "module_file": "torch/__init__.py",
                    "module_sha256": "c" * 64,
                    "files": [{"path": "torch/__init__.py", "sha256": "a" * 64}],
                }
            },
        }
        attestation = RuntimeAttestation.from_payload(payload)
        validate_runtime_attestation(
            attestation,
            state_slice="aligned-holistic-continual-learning-integrated-stack-v2",
            expected_python="3.11.11",
            verify_bytes=False,
        )
        altered = dict(payload)
        altered["python"] = "3.14.5"
        with self.assertRaises(RuntimeAttestationError):
            validate_runtime_attestation(
                RuntimeAttestation.from_payload(altered),
                state_slice="aligned-holistic-continual-learning-integrated-stack-v2",
                expected_python="3.11.11",
            )
        malformed = json.loads(json.dumps(payload))
        malformed["distributions"]["torch"]["files"][0]["sha256"] = "bad"
        with self.assertRaises(RuntimeAttestationError):
            validate_runtime_attestation(
                RuntimeAttestation.from_payload(malformed),
                state_slice="aligned-holistic-continual-learning-integrated-stack-v2",
                expected_python="3.11.11",
            )

    def test_review_packet_digest_is_bound(self):
        packet = build_review_packet(
            config={"state_slice": "aligned-holistic-continual-learning-integrated-stack-v2"},
            upstream_lock={"commit": "source"},
            dependency_lanes={"lane": "modern"},
            corpus_manifest={"manifest_digest": "0" * 64},
            source_digest="0" * 64,
        )
        validate_review_packet(packet)
        packet["source_digest"] = "changed"
        with self.assertRaises(ValueError):
            validate_review_packet(packet)

        incomplete = build_review_packet(
            config={"state_slice": "aligned-holistic-continual-learning-integrated-stack-v2"},
            upstream_lock={"commit": "source"},
            corpus_manifest={"manifest_digest": "0" * 64},
            source_digest="0" * 64,
        )
        with self.assertRaises(ValueError):
            validate_review_packet(incomplete)

    def test_numerical_checks_enforce_noop_and_reach(self):
        self.assertEqual(assert_noop_identity((1.0, 2.0), (1.0, 2.0)), 0.0)
        self.assertEqual(normalized_reconstruction_mse((1.0, 2.0), (1.0, 2.0)), 0.0)
        self.assertGreater(assert_intervention_reach((0.0,), (0.5,)), 0.0)
        with self.assertRaises(CheckError):
            assert_intervention_reach((0.0,), (0.0,))
        with self.assertRaises(CheckError):
            normalized_reconstruction_mse((math.nan,), (0.0,))
        with self.assertRaises(CheckError):
            assert_noop_identity((1.0, 2.0), (1.0,))
        with self.assertRaises(CheckError):
            assert_intervention_reach((0.0,), (1.0,), minimum_delta=-1.0)
        with self.assertRaises(CheckError):
            normalized_reconstruction_mse((), ())

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
        with self.assertRaises(ValueError):
            holm_adjust((math.nan,))
        with self.assertRaises(ValueError):
            holm_adjust(())
        plan = StatisticalPlan()
        power_check(plan)
        self.assertGreaterEqual(achieved_power(plan), plan.power)
        self.assertLess(document_power_lower_bound(plan), plan.joint_power_target)
        with self.assertRaises(ValueError):
            power_check(StatisticalPlan(power_design_status="validated"))

    def test_machine_readable_study_lock_is_frozen(self):
        config = json.loads(Path("configs/integrated-stack-v2.json").read_text(encoding="utf-8"))
        plan = validate_locked_study(config)
        self.assertEqual(plan.as_config()["family_counts"], {"fit": 48, "tune": 48, "assessment": 48})
        altered = dict(config)
        altered["controls"] = list(reversed(altered["controls"]))
        with self.assertRaises(ValueError):
            validate_locked_study(altered)
        altered = dict(config)
        altered["arms"] = list(reversed(altered["arms"]))
        with self.assertRaises(ValueError):
            validate_locked_study(altered)
        altered = json.loads(json.dumps(config))
        del altered["study"]["power_scope"]
        with self.assertRaises(ValueError):
            validate_locked_study(altered)
        altered = json.loads(json.dumps(config))
        altered["study"]["power_design_status"] = "validated"
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
        fit_rows = observations("fit", "fixed_update")
        monitor = RidgeMonitor.fit(fit_rows, ridge=0.1)
        baseline_means = tuple(sum(row.outcomes[index] for row in fit_rows) / len(fit_rows) for index in range(len(AXES)))
        independent_rows = tuple(
            AggregateRow(str(row.family), row.split, 0, row.features, row.outcomes) for row in rows
        )
        reproduced = reproduce_metrics(independent_rows, monitor.predict, baseline_means=baseline_means)
        reproduced_from_packet = reproduce_metrics_from_predictions(
            independent_rows,
            tuple(monitor.predict(row.features) for row in independent_rows),
            baseline_means=baseline_means,
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

    def test_independent_aggregates_reject_roster_corruption(self):
        rows = observations("assessment", "fixed_update")
        fit_rows = observations("fit", "fixed_update")
        monitor = RidgeMonitor.fit(fit_rows, ridge=0.1)
        baseline_means = tuple(sum(row.outcomes[index] for row in fit_rows) / len(fit_rows) for index in range(len(AXES)))
        packet = tuple(
            AggregateRow(str(row.family), row.split, 0, row.features, row.outcomes)
            for row in rows
        )
        with self.assertRaises(AggregateError):
            reproduce_metrics(packet + (packet[0],), monitor.predict, baseline_means=baseline_means)
        mixed = packet[:-1] + (AggregateRow(packet[-1].family_id, "fit", packet[-1].repeat, packet[-1].features, packet[-1].outcomes),)
        with self.assertRaises(AggregateError):
            reproduce_metrics(mixed, monitor.predict, baseline_means=baseline_means)
        expanded = tuple(
            AggregateRow(row.family_id, row.split, repeat, row.features, row.outcomes)
            for row in packet
            for repeat in range(3)
        )
        incomplete = expanded[:-1]
        with self.assertRaises(AggregateError):
            reproduce_metrics(incomplete, monitor.predict, baseline_means=baseline_means)

        with self.assertRaises(AggregateError):
            reproduce_metrics(packet, monitor.predict, baseline_means=())

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

        with self.assertRaises(ReviewError):
            require_independent_accept(
                receipt,
                expected_packet_digest=packet_digest(packet),
                operator_id="operator",
                verify_signature=lambda _receipt, _message: True,
                trusted_public_key_ids=frozenset({"other-key"}),
            )

        self.assertEqual(canonical_review_message("0" * 64), "ahcl.review.packet.v1:" + "0" * 64)

    def test_ed25519_receipt_verifies_canonical_packet_message(self):
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError:
            self.skipTest("cryptography is not installed in the test environment")
        packet = {"state_slice": "aligned-holistic-continual-learning-integrated-stack-v2"}
        locked_digest = packet_digest(packet)
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        signature = private_key.sign(canonical_review_message(locked_digest).encode("ascii"))
        receipt = ReviewReceipt(
            locked_digest,
            "independent-reviewer",
            "ACCEPT",
            signature.hex(),
            public_key_id="reviewer-key-1",
        )
        verifier = Ed25519Verifier({"reviewer-key-1": public_key})
        require_independent_accept(
            receipt,
            expected_packet_digest=locked_digest,
            operator_id="operator",
            verify_signature=verifier,
            trusted_public_key_ids=verifier.trusted_key_ids,
        )

    def test_execution_stays_closed_without_review(self):
        config = {"execution": {"enabled": False}}
        with self.assertRaises(ExecutionClosed):
            require_execution_open(config, None, packet_digest="packet", operator_id="operator")

    def test_enabled_execution_requires_verified_runtime_attestation(self):
        config = {
            "state_slice": "aligned-holistic-continual-learning-integrated-stack-v2",
            "execution": {"enabled": True},
            "runtime": {"python": "3.11.11"},
        }
        packet = {"config": config}
        receipt = ReviewReceipt(packet_digest(packet), "reviewer", "ACCEPT", "signature", public_key_id="key")
        with self.assertRaisesRegex(ExecutionClosed, "runtime distribution attestation"):
            require_execution_open(
                config,
                receipt,
                packet_digest=packet_digest(packet),
                reviewed_packet=packet,
                operator_id="operator",
            )

    def test_lane_attestation_is_unwrapped_only_after_lane_binding(self):
        config = {
            "state_slice": "aligned-holistic-continual-learning-integrated-stack-v2",
            "execution": {"enabled": True},
            "runtime": {"python": "3.11.11"},
        }
        packet = {"config": config}
        receipt = ReviewReceipt(packet_digest(packet), "reviewer", "REJECT", "signature", public_key_id="key")
        runtime = RuntimeAttestation.from_payload({
            "schema": "ahcl.runtime.attestation.v1",
            "state_slice": config["state_slice"],
            "python": "3.11.11",
            "interpreter": {
                "path": str(Path(sys.executable).resolve()),
                "sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
            },
            "distributions": {
                "test": {
                    "distribution": "test",
                    "version": "1.0.0",
                    "import_name": "test",
                    "module_file": "test/__init__.py",
                    "module_sha256": "c" * 64,
                    "files": [{"path": "test/__init__.py", "sha256": "a" * 64}],
                }
            },
        })
        lane = RuntimeLaneAttestation.from_attestation(
            lane="modern", state_slice=config["state_slice"], attestation=runtime
        )
        with patch("ahcl.execution.validate_runtime_lane_attestation") as validate_lane, patch(
            "ahcl.execution.validate_runtime_attestation"
        ):
            with self.assertRaisesRegex(ReviewError, "not ACCEPT"):
                require_execution_open(
                    config,
                    receipt,
                    packet_digest=packet_digest(packet),
                    reviewed_packet=packet,
                    operator_id="operator",
                    runtime_attestation=lane,
                    runtime_lane="modern",
                )
        validate_lane.assert_called_once()


if __name__ == "__main__":
    unittest.main()
