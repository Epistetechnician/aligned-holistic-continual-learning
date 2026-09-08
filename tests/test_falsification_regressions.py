"""State slice: aligned-holistic-continual-learning-integrated-stack-v2."""
import json
import unittest
from pathlib import Path
from ahcl.canonical import digest
from ahcl.preflight import validate_corpus_manifest
from ahcl.stats import holm_adjust, StatisticalPlan
from ahcl.lifecycle import LockedTrial, TrialRow, LifecycleError


class FalsificationRegressions(unittest.TestCase):
    def test_nonfinite_holm_rejected(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                holm_adjust((value,))

    def test_rehashed_duplicate_document_rejected(self):
        manifest = json.loads(Path('configs/corpus-manifest.v2.json').read_text())
        manifest['documents'][1]['id'] = manifest['documents'][0]['id']
        manifest['manifest_digest'] = digest({k:v for k,v in manifest.items() if k != 'manifest_digest'})
        with self.assertRaises(ValueError):
            validate_corpus_manifest(manifest)

    def locked_trial(self):
        trial = LockedTrial({'nested': {'x': 1}}, StatisticalPlan())
        rows = lambda split: [TrialRow(f'{split}-{i}', split, r, (), ())
                              for i in range(48) for r in range(3)]
        trial.fit(rows('fit'), lambda data: (object(), {'coefficient': 1}))
        trial.tune(rows('tune'), lambda predictor,data: {'metric': 0})
        return trial, rows('assessment')

    def test_assessment_is_one_use(self):
        trial, rows = self.locked_trial()
        trial.assess(rows, lambda predictor,data: {'metric': 0})
        with self.assertRaises(LifecycleError):
            trial.assess(rows, lambda predictor,data: {'metric': 0})

    def test_failed_assessment_cannot_retry(self):
        trial, rows = self.locked_trial()
        with self.assertRaises(LifecycleError):
            trial.assess(rows[:-1], lambda predictor,data: {'metric': 0})
        with self.assertRaises(LifecycleError):
            trial.assess(rows, lambda predictor,data: {'metric': 0})

    def test_config_mutation_rejected_before_reading_assessment(self):
        trial, rows = self.locked_trial()
        trial.config['nested']['x'] = 2
        def forbidden():
            raise AssertionError('assessment iterator was accessed')
            yield
        with self.assertRaises(LifecycleError):
            trial.assess(forbidden(), lambda predictor,data: {'metric': 0})
