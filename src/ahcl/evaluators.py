"""Behavioral endpoints for model-backed continual-learning trials.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any, Sequence


def _logits(output: Any) -> Any:
    return output.logits if hasattr(output, "logits") else output


def next_token_cross_entropy(logits: Any, labels: Any) -> float:
    torch = __import__("torch")
    functional = torch.nn.functional
    value = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    return float(value.detach().cpu())


def exact_match_accuracy(logits: Any, labels: Any) -> float:
    predictions = logits.argmax(dim=-1)
    valid = labels != -100
    total = int(valid.sum().item())
    if total == 0:
        raise ValueError("labels contain no scored positions")
    return float(((predictions == labels) & valid).sum().item() / total)


def expected_calibration_error(logits: Any, labels: Any, bins: int = 10) -> float:
    if bins < 2:
        raise ValueError("bins must be at least two")
    torch = __import__("torch")
    probabilities = torch.softmax(logits, dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    valid = labels != -100
    confidence = confidence[valid]
    correctness = (predictions[valid] == labels[valid]).float()
    if confidence.numel() == 0:
        raise ValueError("labels contain no scored positions")
    error = torch.zeros((), device=confidence.device)
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (confidence >= low) & (confidence <= high if index == bins - 1 else confidence < high)
        if mask.any():
            error = error + mask.float().mean() * (confidence[mask].mean() - correctness[mask].mean()).abs()
    return float(error.detach().cpu())


def policy_violation_rate(logits: Any, forbidden_token_ids: Sequence[int]) -> float:
    if not forbidden_token_ids:
        raise ValueError("forbidden_token_ids must not be empty")
    predictions = logits.argmax(dim=-1)
    forbidden = predictions.new_tensor(list(forbidden_token_ids))
    return float(torch_any(predictions.unsqueeze(-1) == forbidden).float().mean().item())


def torch_any(value: Any) -> Any:
    torch = __import__("torch")
    return torch.any(value, dim=-1)


class TorchBehaviorEvaluator:
    """Evaluate loss, accuracy, calibration, and policy violations."""

    def __init__(self, model: Any, *, forbidden_token_ids: Sequence[int]):
        self.model = model
        self.forbidden_token_ids = tuple(forbidden_token_ids)

    def __call__(self, batch: Any) -> dict[str, float]:
        self.model.eval()
        with __import__("torch").no_grad():
            output = self.model(input_ids=batch.input_ids, labels=batch.labels)
        logits = _logits(output)
        return {
            "next_token_cross_entropy": next_token_cross_entropy(logits, batch.labels),
            "exact_match_accuracy": exact_match_accuracy(logits, batch.labels),
            "expected_calibration_error": expected_calibration_error(logits, batch.labels),
            "policy_violation_rate": policy_violation_rate(logits, self.forbidden_token_ids),
        }
