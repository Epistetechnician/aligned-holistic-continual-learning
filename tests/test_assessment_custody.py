"""Assessment attempt custody tests.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from ahcl.assessment_custody import AssessmentAttemptError, claim_assessment_attempt


PACKET = "a" * 64
PREDICTOR = "b" * 64


def custody_root(parent: Path) -> Path:
    root = parent / "custody"
    root.mkdir()
    root.chmod(0o700)
    return root


class AssessmentCustodyTests(unittest.TestCase):
    def test_claim_is_bound_and_survives_restart(self):
        with TemporaryDirectory() as directory:
            root = custody_root(Path(directory))
            claim = claim_assessment_attempt(root, packet_id=PACKET, predictor_id=PREDICTOR)
            self.assertTrue(claim.path.exists())
            self.assertEqual(
                json.loads(claim.path.read_text()),
                {
                    "packet_id": PACKET,
                    "predictor_id": PREDICTOR,
                    "schema": "ahcl.assessment-attempt-claim.v1",
                    "state_slice": "aligned-holistic-continual-learning-integrated-stack-v2",
                },
            )
            with self.assertRaisesRegex(AssessmentAttemptError, "already consumed"):
                claim_assessment_attempt(root, packet_id=PACKET, predictor_id=PREDICTOR)

    def test_failed_attempt_marker_is_still_consumed(self):
        with TemporaryDirectory() as directory:
            root = custody_root(Path(directory))
            marker = root / f"assessment-attempt.{PACKET}.{PREDICTOR}.json"
            marker.touch(mode=0o600)
            with self.assertRaisesRegex(AssessmentAttemptError, "already consumed"):
                claim_assessment_attempt(root, packet_id=PACKET, predictor_id=PREDICTOR)

    def test_concurrent_claim_has_one_winner(self):
        with TemporaryDirectory() as directory:
            root = custody_root(Path(directory))
            with ThreadPoolExecutor(max_workers=8) as pool:
                outcomes = list(pool.map(lambda _: self._claim(root), range(8)))
            self.assertEqual(sum(outcomes), 1)

    @staticmethod
    def _claim(root: Path) -> bool:
        try:
            claim_assessment_attempt(root, packet_id=PACKET, predictor_id=PREDICTOR)
            return True
        except AssessmentAttemptError:
            return False

    def test_malformed_identity_rejected(self):
        values = ["", "A" * 64, "a" * 63, "../" + "a" * 62, 7, None]
        with TemporaryDirectory() as directory:
            root = custody_root(Path(directory))
            for value in values:
                with self.subTest(value=value), self.assertRaisesRegex(AssessmentAttemptError, "canonical"):
                    claim_assessment_attempt(root, packet_id=value, predictor_id=PREDICTOR)

    def test_non_owner_only_root_rejected(self):
        with TemporaryDirectory() as directory:
            root = custody_root(Path(directory))
            root.chmod(0o755)
            with self.assertRaisesRegex(AssessmentAttemptError, "0700"):
                claim_assessment_attempt(root, packet_id=PACKET, predictor_id=PREDICTOR)

    def test_symlink_root_and_marker_rejected(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            real = custody_root(parent)
            link = parent / "root-link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(AssessmentAttemptError, "real directory"):
                claim_assessment_attempt(link, packet_id=PACKET, predictor_id=PREDICTOR)

            marker = real / f"assessment-attempt.{PACKET}.{PREDICTOR}.json"
            target = real / "target"
            target.touch()
            marker.symlink_to(target)
            with self.assertRaisesRegex(AssessmentAttemptError, "symlink"):
                claim_assessment_attempt(real, packet_id=PACKET, predictor_id=PREDICTOR)


if __name__ == "__main__":
    unittest.main()
