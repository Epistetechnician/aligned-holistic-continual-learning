"""Runtime distribution attestation tests.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

import hashlib
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from ahcl.canonical import canonical_bytes
from ahcl.runtime_attestation import (
    RuntimeAttestation,
    RuntimeAttestationError,
    RuntimeLaneAttestation,
    RuntimeRequirement,
    attest_lane,
    attest_runtime,
    main,
    validate_runtime_lane_attestation,
    validate_runtime_attestation,
)


STATE_SLICE = "aligned-holistic-continual-learning-integrated-stack-v2"


def valid_payload() -> dict:
    interpreter = Path(sys.executable).resolve()
    return {
        "schema": "ahcl.runtime.attestation.v1",
        "state_slice": STATE_SLICE,
        "python": "3.11.11",
        "interpreter": {
            "path": str(interpreter),
            "sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest(),
        },
        "distributions": {
            "example": {
                "distribution": "example",
                "version": "1.2.3",
                "import_name": "example",
                "files_root": str(Path(sys.executable).resolve().parent.parent),
                "module_file": "example/__init__.py",
                "module_sha256": "c" * 64,
                "files": [{"path": "example/__init__.py", "sha256": "a" * 64}],
            }
        },
    }


class RuntimeAttestationTests(unittest.TestCase):
    def test_attest_runtime_hashes_distribution_files_and_module_location(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "example"
            package.mkdir()
            module_file = package / "__init__.py"
            module_file.write_bytes(b"example-runtime")
            interpreter = root / "python3.11"
            interpreter.write_bytes(b"python-runtime")
            metadata = root / "example-1.2.3.dist-info"
            metadata.mkdir()
            metadata_file = metadata / "METADATA"
            metadata_file.write_bytes(b"Name: example\nVersion: 1.2.3\n")

            class Distribution:
                version = "1.2.3"
                files = [Path("example/__init__.py"), Path("example-1.2.3.dist-info/METADATA")]

                @staticmethod
                def locate_file(entry):
                    return root / entry

            module = SimpleNamespace(__file__=str(module_file))
            requirement = RuntimeRequirement("example", "example", "1.2.3", root)
            with patch("ahcl.runtime_attestation.platform.python_version", return_value="3.11.11"), patch(
                "ahcl.runtime_attestation.sys.executable", str(interpreter)
            ), patch(
                "ahcl.runtime_attestation.importlib.metadata.distribution", return_value=Distribution()
            ), patch("ahcl.runtime_attestation.importlib.import_module", return_value=module):
                attestation = attest_runtime(
                    state_slice=STATE_SLICE,
                    expected_python="3.11.11",
                    requirements=(requirement,),
                )
                validate_runtime_attestation(
                    attestation,
                    state_slice=STATE_SLICE,
                    expected_python="3.11.11",
                )
                tampered = attestation.payload
                tampered["distributions"]["example"]["files"][0]["sha256"] = "0" * 64
                with self.assertRaisesRegex(RuntimeAttestationError, "file bytes changed"):
                    validate_runtime_attestation(
                        RuntimeAttestation.from_payload(tampered),
                        state_slice=STATE_SLICE,
                        expected_python="3.11.11",
                    )

            self.assertEqual(attestation.payload["python"], "3.11.11")
            self.assertEqual(attestation.payload["interpreter"]["path"], str(interpreter.resolve()))
            self.assertEqual(
                attestation.payload["interpreter"]["sha256"],
                hashlib.sha256(interpreter.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                attestation.payload["distributions"]["example"]["module_sha256"],
                hashlib.sha256(module_file.read_bytes()).hexdigest(),
            )
            self.assertEqual(attestation.payload["distributions"]["example"]["version"], "1.2.3")
            files = attestation.payload["distributions"]["example"]["files"]
            self.assertEqual(
                next(item["sha256"] for item in files if item["path"] == "example/__init__.py"),
                hashlib.sha256(module_file.read_bytes()).hexdigest(),
            )
            self.assertEqual(len(attestation.attestation_digest), 64)

    def test_attestation_is_immutable_and_schema_bound(self):
        attestation = RuntimeAttestation.from_payload(valid_payload())
        original = attestation.attestation_digest
        payload = attestation.payload
        payload["python"] = "3.11.12"
        self.assertEqual(attestation.attestation_digest, original)
        with self.assertRaisesRegex(RuntimeAttestationError, "Python identity"):
            validate_runtime_attestation(attestation, state_slice=STATE_SLICE, expected_python="3.11.12")

        malformed = valid_payload()
        malformed["distributions"]["example"]["files"][0]["sha256"] = "not-a-hash"
        with self.assertRaisesRegex(RuntimeAttestationError, "hash"):
            validate_runtime_attestation(
                RuntimeAttestation.from_payload(malformed),
                state_slice=STATE_SLICE,
                expected_python="3.11.11",
            )
        altered_interpreter = valid_payload()
        altered_interpreter["interpreter"]["sha256"] = "d" * 64
        with self.assertRaisesRegex(RuntimeAttestationError, "interpreter does not match"):
            validate_runtime_attestation(
                RuntimeAttestation.from_payload(altered_interpreter),
                state_slice=STATE_SLICE,
                expected_python="3.11.11",
            )

    def test_attestation_rejects_wrong_state_and_python_before_package_access(self):
        with self.assertRaisesRegex(RuntimeAttestationError, "Python version mismatch"):
            attest_runtime(
                state_slice=STATE_SLICE,
                expected_python="3.11.11",
                requirements=(RuntimeRequirement("example", "example", "1.2.3", Path(".")),),
            )

        with self.assertRaisesRegex(RuntimeAttestationError, "state slice"):
            validate_runtime_attestation(
                RuntimeAttestation.from_payload(valid_payload()),
                state_slice="other-state-slice",
                expected_python="3.11.11",
            )

    def test_symlink_distribution_file_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"target")
            link = root / "link"
            link.symlink_to(target)

            class Distribution:
                version = "1.2.3"
                files = [Path("link")]

                @staticmethod
                def locate_file(entry):
                    return root / entry

            with patch("ahcl.runtime_attestation.platform.python_version", return_value="3.11.11"), patch(
                "ahcl.runtime_attestation.importlib.metadata.distribution", return_value=Distribution()
            ):
                with self.assertRaisesRegex(RuntimeAttestationError, "symlink"):
                    attest_runtime(
                        state_slice=STATE_SLICE,
                        expected_python="3.11.11",
                        requirements=(RuntimeRequirement("example", "example", "1.2.3", root),),
                    )

    def test_missing_interpreter_is_rejected(self):
        with patch("ahcl.runtime_attestation.platform.python_version", return_value="3.11.11"), patch(
            "ahcl.runtime_attestation.sys.executable", "/missing/python3.11"
        ):
            with self.assertRaisesRegex(RuntimeAttestationError, "interpreter"):
                attest_runtime(
                    state_slice=STATE_SLICE,
                    expected_python="3.11.11",
                    requirements=(RuntimeRequirement("example", "example", "1.2.3", Path(".")),),
                )

    def test_attest_lane_derives_pins_and_runs_only_in_operator_environment(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lanes_path = root / "lanes.json"
            config_path = root / "config.json"
            lanes_path.write_text(
                json.dumps(
                    {
                        "schema": "ahcl.dependency.lanes.v1",
                        "state_slice": STATE_SLICE,
                        "lanes": {
                            "modern": {
                                "constraints": {
                                    "python": "3.11.11",
                                    "torch": "==2.6.0",
                                    "transformers": "==5.9.0",
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "schema": "ahcl.integrated.stack.config.v1",
                        "state_slice": STATE_SLICE,
                        "runtime": {"python": "3.11.11", "torch": "2.6.0", "transformers": "5.9.0"},
                    }
                ),
                encoding="utf-8",
            )
            environment = root / "environment"
            (environment / "bin").mkdir(parents=True)
            interpreter = environment / "bin/python"
            interpreter.write_bytes(b"synthetic-python-3.11.11")
            runtime = RuntimeAttestation.from_payload(
                {
                    "schema": "ahcl.runtime.attestation.v1",
                    "state_slice": STATE_SLICE,
                    "python": "3.11.11",
                    "interpreter": {"path": str(interpreter), "sha256": "a" * 64},
                    "distributions": {
                        "torch": {
                            "distribution": "torch",
                            "version": "2.6.0",
                            "import_name": "torch",
                            "files_root": str(environment.resolve()),
                            "module_file": "torch/__init__.py",
                            "module_sha256": "b" * 64,
                            "files": [{"path": "torch/__init__.py", "sha256": "c" * 64}],
                        },
                        "transformers": {
                            "distribution": "transformers",
                            "version": "5.9.0",
                            "import_name": "transformers",
                            "files_root": str(environment.resolve()),
                            "module_file": "transformers/__init__.py",
                            "module_sha256": "d" * 64,
                            "files": [{"path": "transformers/__init__.py", "sha256": "e" * 64}],
                        },
                    },
                }
            )
            child_document = json.dumps(runtime.document(), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            with patch("ahcl.runtime_attestation.subprocess.run") as run:
                run.return_value = SimpleNamespace(returncode=0, stdout=child_document, stderr=b"")
                result = attest_lane(
                    lane="modern",
                    lanes_path=lanes_path,
                    integrated_config_path=config_path,
                    environment=environment,
                )

            self.assertIsInstance(result, RuntimeLaneAttestation)
            self.assertEqual(result.payload["lane"], "modern")
            self.assertEqual(result.payload["runtime"]["python"], "3.11.11")
            call = run.call_args
            self.assertEqual(call.args[0][0], str(interpreter.resolve()))
            request = json.loads(call.kwargs["input"].decode("utf-8"))
            self.assertEqual(
                [item["distribution"] for item in request["requirements"]],
                ["torch", "transformers"],
            )
            self.assertEqual(request["source_root"], str(environment.resolve()))
            self.assertEqual(call.kwargs["env"]["PYTHONNOUSERSITE"], "1")
            self.assertEqual(result._canonical_payload, canonical_bytes(result.payload))

    def test_attest_lane_rejects_unpinned_lane_before_operator_process(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lanes_path = root / "lanes.json"
            config_path = root / "config.json"
            lanes_path.write_text(
                json.dumps(
                    {
                        "schema": "ahcl.dependency.lanes.v1",
                        "state_slice": STATE_SLICE,
                        "lanes": {"visualization": {"network": "disabled_until_review"}},
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {"schema": "ahcl.integrated.stack.config.v1", "state_slice": STATE_SLICE, "runtime": {}}
                ),
                encoding="utf-8",
            )
            with patch("ahcl.runtime_attestation.subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeAttestationError, "no pinned constraints"):
                    attest_lane(
                        lane="visualization",
                        lanes_path=lanes_path,
                        integrated_config_path=config_path,
                        interpreter=root / "python",
                    )
            run.assert_not_called()

    def test_lane_document_digest_and_cli_output_are_canonical(self):
        runtime = RuntimeAttestation.from_payload(valid_payload())
        lane = RuntimeLaneAttestation.from_attestation(
            lane="modern", state_slice=STATE_SLICE, attestation=runtime
        )
        validate_runtime_lane_attestation(
            lane, state_slice=STATE_SLICE, lane="modern", expected_python="3.11.11"
        )
        output = io.StringIO()
        with patch("ahcl.runtime_attestation.attest_lane", return_value=lane), redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "--lane",
                        "modern",
                        "--environment",
                        "/operator/env",
                        "--import-name",
                        "transformer-lens=transformer_lens",
                    ]
                ),
                0,
            )
        encoded = output.getvalue().encode("utf-8").rstrip(b"\n")
        self.assertEqual(encoded, json.dumps(lane.document(), sort_keys=True, separators=(",", ":")).encode("utf-8"))
        self.assertEqual(len(lane.lane_attestation_digest), 64)
        round_trip = RuntimeLaneAttestation.from_document(lane.document())
        self.assertEqual(round_trip._canonical_payload, lane._canonical_payload)
        self.assertEqual(round_trip.runtime_attestation.attestation_digest, runtime.attestation_digest)

    def test_runtime_document_round_trip_rejects_digest_tampering(self):
        document = RuntimeAttestation.from_payload(valid_payload()).document()
        document["attestation_digest"] = "0" * 64
        with self.assertRaisesRegex(RuntimeAttestationError, "digest mismatch"):
            RuntimeAttestation.from_document(document)

    def test_runtime_json_rejects_duplicate_object_keys(self):
        with self.assertRaisesRegex(RuntimeAttestationError, "malformed"):
            RuntimeAttestation.from_json(
                b'{"schema":"ahcl.runtime.attestation.v1","schema":"duplicate"}'
            )


if __name__ == "__main__":
    unittest.main()
