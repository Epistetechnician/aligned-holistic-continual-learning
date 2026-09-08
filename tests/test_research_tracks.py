"""State slice: aligned-holistic-continual-learning-integrated-stack-v2."""
import unittest
from ahcl.research_tracks import (HorizonPoint, feature_stability, horizon_summary,
                                 selective_repair, alternative_predictor_features)


class ResearchTracksTests(unittest.TestCase):
    def test_feature_effect_reversal_is_detected(self):
        result = feature_stability((1., -2.), (-1., 2.))
        self.assertAlmostEqual(result['effect_cosine'], -1)
        self.assertEqual(result['sign_agreement'], 0)

    def test_reject_all_cannot_pass_learning_floor(self):
        result = horizon_summary([HorizonPoint(1, 0, 0), HorizonPoint(2, 0, 0)], minimum_learning=.1)
        self.assertFalse(result['learning_floor_pass'])

    def test_final_recovery_does_not_erase_earlier_harm(self):
        result = horizon_summary([HorizonPoint(1, .2, .8), HorizonPoint(2, .2, 0)], minimum_learning=.1)
        self.assertEqual(result['cumulative_positive_harm'], .8)
        self.assertEqual(result['final_harm'], 0)

    def test_missing_checkpoint_fails(self):
        with self.assertRaises(ValueError):
            horizon_summary([HorizonPoint(2, .2, 0)], minimum_learning=.1)

    def test_repair_must_preserve_new_learning(self):
        result = selective_repair(old_base_loss=1, old_damaged_loss=2, old_repaired_loss=1,
                                  new_updated_loss=1, new_repaired_loss=2,
                                  minimum_recovery=.5, maximum_new_loss_increase=.1)
        self.assertFalse(result['pass'])

    def test_alternative_inputs_are_hand_calculable(self):
        self.assertEqual(alternative_predictor_features((3.,4.), (0.,2.), 2, 1, (1.,0.), (0.,1.)),
                         (5.,2.,-1,0))

    def test_missing_and_nonfinite_effects_fail(self):
        for a,b in [((),()), ((0.,),(0.,)), ((float('nan'),),(1.,))]:
            with self.assertRaises(ValueError):
                feature_stability(a,b)
