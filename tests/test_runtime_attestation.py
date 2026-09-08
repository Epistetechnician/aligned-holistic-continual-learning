"""Runtime distribution attestation tests.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from ahcl.runtime_attestation import (
    RuntimeAttestation,
    RuntimeAttestationError,
    RuntimeRequirement,
    attest_runtime,
    validate_runtime_attestation,
)


STATE_SLICE = "aligned-holistic-continual-learning-integrated-stack-v2"


def valid_payload() -> dict:
    return {
        "schema": "ahcl.runtime.attestation.v1",
        "state_slice": STATE_SLICE,
        "python": "3.11.11",
        "interpreter": {"path": "/opt/python/bin/python3.11", "sha256": "b" * 64},
        "distributions": {
            "example": {
                "distribution": "example",
                "version": "1.2.3",
                "import_name": "example",
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
            validate_runtime_attestation(
                attestation,
                state_slice=STATE_SLICE,
                expected_python="3.11.11",
            )

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


if __name__ == "__main__":
    unittest.main()
