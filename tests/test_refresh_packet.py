"""Review packet refresh tests.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ahcl.preflight import PreflightError, source_tree_digest
from ahcl.refresh_packet import build_current_packet


class ReviewPacketRefreshTests(unittest.TestCase):
    def test_current_packet_builder_binds_root_gate_manifests(self):
        root = Path(__file__).parents[1]
        packet = build_current_packet(root)
        self.assertEqual(packet["config"]["state_slice"], "aligned-holistic-continual-learning-integrated-stack-v2")
        self.assertEqual(len(packet["source_digest"]), 64)

    def test_tracked_packet_matches_current_public_inputs(self):
        root = Path(__file__).parents[1]
        packet = json.loads((root / "configs/review-packet.v2.json").read_text(encoding="utf-8"))
        self.assertEqual(packet, build_current_packet(root))

    def test_source_digest_rejects_symlinked_authored_bytes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            link = root / "src" / "external_link"
            target = root / "external_bytes"
            target.write_bytes(b"external")
            link.symlink_to(target)
            with self.assertRaisesRegex(PreflightError, "symlink"):
                source_tree_digest(root)


if __name__ == "__main__":
    unittest.main()
