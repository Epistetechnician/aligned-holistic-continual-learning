"""State slice: aligned-holistic-continual-learning-integrated-stack-v2.

Ephemeral test keys prove verifier mechanics only; never execution acceptance.
"""
import unittest
import hashlib
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from ahcl.canonical import digest
from ahcl.verified_review import TrustedReviewer, signed_message, verify_launch_receipt
from ahcl.sealed_predictor import SealedLinearPredictor
from ahcl.assessment_boundary import assess_once
from ahcl.runtime_attestation import RuntimeAttestation, RuntimeLaneAttestation


class VerificationTests(unittest.TestCase):
    def setUp(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        self.key = Ed25519PrivateKey.generate()
        self.registry = {'test': TrustedReviewer('reviewer', self.key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))}
        self.packet = {'schema':'ahcl.review.packet.v1', 'state_slice':'test', 'config':{'state_slice':'test'},
                       **{k:'a'*64 for k in ('source_digest','upstream_lock_digest','dependency_lanes_digest','corpus_manifest_digest')}}
        self.packet['packet_digest'] = digest(self.packet)
        self.receipt = {'packet_digest':self.packet['packet_digest'], 'reviewer_id':'reviewer',
                        'public_key_id':'test', 'decision':'ACCEPT', 'purpose':'qualification'}
        self.receipt['signature'] = self.key.sign(signed_message(self.packet['packet_digest'],'reviewer','test','qualification')).hex()

    def verify(self, receipt=None, registry=None, operator='operator'):
        return verify_launch_receipt(self.packet,self.packet['config'],receipt or self.receipt,
                                    self.registry if registry is None else registry,
                                    operator_id=operator,purpose='qualification')

    def test_real_signature_and_tampering(self):
        self.assertEqual(len(self.verify()),64)
        bad = deepcopy(self.receipt)
        bad['signature'] = '00'*64
        with self.assertRaises(ValueError): self.verify(bad)

    def test_unknown_key_self_review_and_identity_substitution(self):
        with self.assertRaises(ValueError): self.verify(registry={})
        with self.assertRaises(ValueError): self.verify(operator='reviewer')
        bad = deepcopy(self.receipt)
        bad['reviewer_id'] = 'someone-else'
        with self.assertRaises(ValueError): self.verify(bad)

    def test_predictor_detaches_mutable_inputs(self):
        weights = [[2.]]
        predictor = SealedLinearPredictor(weights,[1.])
        before = predictor.identity
        weights[0][0] = 9.
        self.assertEqual(predictor.predict([3.]),(7.,))
        self.assertEqual(before,predictor.identity)
        self.assertNotEqual(before,SealedLinearPredictor([[9.]],[1.]).identity)

    def test_boundary_consumes_failed_callback_and_binds_predictor(self):
        predictor = SealedLinearPredictor([[2.]],[1.])
        self.packet['config'].update(
            execution={'enabled':True},
            assessment={'predictor_digest':predictor.identity},
            runtime={'python':'3.11.11'},
        )
        self.packet['packet_digest'] = digest({k:v for k,v in self.packet.items() if k != 'packet_digest'})
        self.receipt.update(packet_digest=self.packet['packet_digest'], purpose='assessment')
        self.receipt['signature'] = self.key.sign(signed_message(self.packet['packet_digest'],'reviewer','test','assessment')).hex()
        called = []
        def fail(p):
            called.append(p.identity)
            raise RuntimeError('simulated evaluator failure')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            runtime_attestation = RuntimeAttestation.from_payload({
                'schema':'ahcl.runtime.attestation.v1',
                'state_slice':'test',
                'python':'3.11.11',
                'interpreter': {
                    'path': str(Path(sys.executable).resolve()),
                    'sha256': hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
                },
                'distributions': {
                    'test': {
                        'distribution':'test',
                        'version':'1.0.0',
                        'import_name':'test',
                        'files_root': str(Path(sys.executable).resolve().parent.parent),
                        'module_file':'test/__init__.py',
                        'module_sha256':'c'*64,
                        'files':[{'path':'test/__init__.py', 'sha256':'a'*64}],
                    }
                },
            })
            runtime_lane = RuntimeLaneAttestation.from_attestation(
                lane='modern', state_slice='test', attestation=runtime_attestation
            )
            kwargs = dict(packet=self.packet, config=self.packet['config'], receipt=self.receipt,
                          registry=self.registry, operator_id='operator', custody_root=root,
                          predictor=predictor, evaluate=fail, runtime_attestation=runtime_lane,
                          runtime_lane='modern')
            with patch("ahcl.assessment_boundary.validate_runtime_lane_attestation") as validate_lane, \
                    patch("ahcl.assessment_boundary.validate_runtime_attestation"):
                with self.assertRaises(RuntimeError): assess_once(**kwargs)
                with self.assertRaises(ValueError): assess_once(**kwargs)
            validate_lane.assert_called()
            self.assertEqual(called,[predictor.identity])
