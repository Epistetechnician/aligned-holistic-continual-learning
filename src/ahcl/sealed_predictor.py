"""Fixed linear predictor whose serialized parameters are its computation.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
No arbitrary callback, pickle, or external mutable model is accepted.
"""
from dataclasses import dataclass
import math
from .canonical import digest


@dataclass(frozen=True)
class SealedLinearPredictor:
    weights: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]

    def __post_init__(self):
        weights = tuple(tuple(float(x) for x in row) for row in self.weights)
        intercepts = tuple(float(x) for x in self.intercepts)
        if not weights or not weights[0] or len(weights) != len(intercepts):
            raise ValueError('predictor dimensions must be nonempty and aligned')
        if any(len(row) != len(weights[0]) for row in weights):
            raise ValueError('ragged predictor')
        if any(not math.isfinite(x) for row in weights for x in row) or any(not math.isfinite(x) for x in intercepts):
            raise ValueError('nonfinite predictor')
        object.__setattr__(self, 'weights', weights)
        object.__setattr__(self, 'intercepts', intercepts)

    def payload(self):
        return {'schema': 'ahcl.linear-predictor.v1', 'weights': self.weights,
                'intercepts': self.intercepts, 'feature_transform': 'identity'}

    @property
    def identity(self):
        return digest(self.payload())

    def predict(self, features):
        values = tuple(float(x) for x in features)
        if len(values) != len(self.weights[0]) or any(not math.isfinite(x) for x in values):
            raise ValueError('invalid predictor input')
        result = tuple(b + sum(w*x for w,x in zip(row,values)) for row,b in zip(self.weights,self.intercepts))
        if any(not math.isfinite(x) for x in result):
            raise ValueError('prediction overflow')
        return result
