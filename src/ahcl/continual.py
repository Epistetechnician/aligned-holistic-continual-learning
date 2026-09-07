"""Model-agnostic continual-learning update loop with lazy PyTorch use.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol


@dataclass(frozen=True)
class StreamBatch:
    batch_id: str
    input_ids: Any
    labels: Any
    metadata: Mapping[str, Any]


class UpdatePolicy(Protocol):
    def update(self, batch: StreamBatch) -> float:
        """Apply one bounded update and return its scalar loss."""


class BatchEvaluator(Protocol):
    def __call__(self, batch: StreamBatch) -> Mapping[str, float]:
        """Measure behavior without changing model state."""


class TorchGradientUpdatePolicy:
    """Apply ordinary gradient updates while keeping the update boundary explicit."""

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Callable[[Any, StreamBatch], Any],
        *,
        ewc_reference: Mapping[str, Any] | None = None,
        ewc_fisher: Mapping[str, Any] | None = None,
        ewc_lambda: float = 0.0,
    ):
        if ewc_lambda < 0.0:
            raise ValueError("ewc_lambda must be non-negative")
        if (ewc_reference is None) != (ewc_fisher is None):
            raise ValueError("ewc_reference and ewc_fisher must be supplied together")
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.ewc_reference = ewc_reference
        self.ewc_fisher = ewc_fisher
        self.ewc_lambda = ewc_lambda

    def update(self, batch: StreamBatch) -> float:
        torch = __import__("torch")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model, batch)
        if self.ewc_reference is not None and self.ewc_fisher is not None and self.ewc_lambda:
            penalty = torch.zeros((), device=loss.device)
            for name, parameter in self.model.named_parameters():
                if name in self.ewc_reference and name in self.ewc_fisher:
                    penalty = penalty + (self.ewc_fisher[name] * (parameter - self.ewc_reference[name]).pow(2)).sum()
            loss = loss + 0.5 * self.ewc_lambda * penalty
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())


class ContinualStreamRunner:
    """Run ordered stream updates and evaluation checkpoints."""

    def __init__(self, policy: UpdatePolicy, evaluator: BatchEvaluator, *, evaluate_every: int = 1):
        if evaluate_every < 1:
            raise ValueError("evaluate_every must be positive")
        self.policy = policy
        self.evaluator = evaluator
        self.evaluate_every = evaluate_every

    def run(self, stream: Iterable[StreamBatch]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, batch in enumerate(stream, start=1):
            loss = self.policy.update(batch)
            record: dict[str, Any] = {"batch_id": batch.batch_id, "update_loss": loss}
            if index % self.evaluate_every == 0:
                record["metrics"] = dict(self.evaluator(batch))
            records.append(record)
        return records
