"""Model-free measurements for the parallel research protocols.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
These functions summarize supplied measurements; they do not execute models.
"""

import math
from dataclasses import dataclass
from typing import Sequence


def _finite(values: Sequence[float]) -> None:
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("measurements must be nonempty and finite")


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    _finite(left)
    _finite(right)
    if len(left) != len(right):
        raise ValueError("dimensions differ")
    scale = math.sqrt(sum(x*x for x in left)) * math.sqrt(sum(x*x for x in right))
    if scale == 0:
        raise ValueError("zero vectors have undefined cosine similarity")
    return max(-1.0, min(1.0, sum(x*y for x, y in zip(left, right))/scale))


def feature_stability(before: Sequence[float], after: Sequence[float]) -> dict[str, float]:
    """Compare fixed-ID causal effect vectors, without post-hoc rematching."""
    cosine = cosine_similarity(before, after)
    return {"effect_cosine": cosine,
            "effect_rmse": math.sqrt(sum((x-y)**2 for x,y in zip(before, after))/len(before)),
            "sign_agreement": sum((x > 0)-(x < 0) == (y > 0)-(y < 0)
                                  for x,y in zip(before, after))/len(before)}


def alternative_predictor_features(gradient: Sequence[float], update: Sequence[float],
                                 loss_before: float, loss_after: float,
                                 representation_before: Sequence[float],
                                 representation_after: Sequence[float]) -> tuple[float, ...]:
    """Candidate-probe inputs only; callers must exclude assessment outcomes."""
    _finite(gradient)
    _finite(update)
    _finite((loss_before, loss_after))
    return (math.sqrt(sum(x*x for x in gradient)), math.sqrt(sum(x*x for x in update)),
            loss_after-loss_before, cosine_similarity(representation_before, representation_after))


@dataclass(frozen=True)
class HorizonPoint:
    step: int
    incoming_gain: float
    protected_loss_increase: float


def horizon_summary(points: Sequence[HorizonPoint], *, minimum_learning: float) -> dict:
    """Report cumulative positive harm at every equally spaced checkpoint."""
    if not points or [p.step for p in points] != list(range(1, len(points)+1)):
        raise ValueError("complete ordered checkpoints starting at one are required")
    _finite((minimum_learning,))
    if minimum_learning <= 0:
        raise ValueError("a positive learning floor is required")
    _finite([x for p in points for x in (p.incoming_gain, p.protected_loss_increase)])
    harms = [max(0.0, p.protected_loss_increase) for p in points]
    gain = sum(p.incoming_gain for p in points)/len(points)
    return {"mean_incoming_gain": gain, "learning_floor_pass": gain >= minimum_learning,
            "cumulative_positive_harm": sum(harms), "peak_harm": max(harms),
            "final_harm": harms[-1], "recovery_from_peak": max(harms)-harms[-1]}


def selective_repair(*, old_base_loss: float, old_damaged_loss: float,
                     old_repaired_loss: float, new_updated_loss: float,
                     new_repaired_loss: float, minimum_recovery: float,
                     maximum_new_loss_increase: float) -> dict:
    """Require real damage, old-task recovery, and preservation of new learning."""
    _finite((old_base_loss, old_damaged_loss, old_repaired_loss, new_updated_loss,
             new_repaired_loss, minimum_recovery, maximum_new_loss_increase))
    damage = old_damaged_loss-old_base_loss
    if damage <= 0 or not 0 < minimum_recovery <= 1 or maximum_new_loss_increase < 0:
        raise ValueError("repair requires positive damage and valid frozen thresholds")
    recovery = (old_damaged_loss-old_repaired_loss)/damage
    new_cost = new_repaired_loss-new_updated_loss
    return {"recovered_fraction": recovery, "new_loss_increase": new_cost,
            "pass": recovery >= minimum_recovery and new_cost <= maximum_new_loss_increase}
