"""Behavioral endpoints for model-backed continual-learning trials.

State slice: aligned-holistic-continual-learning-integrated-stack-v2.
"""

from __future__ import annotations

from typing import Any, Sequence


def _logits(output: Any) -> Any:
    return output.logits if hasattr(output, "logits") else output


def align_next_tokens(logits: Any, labels: Any, attention_mask: Any) -> tuple[Any, Any]:
    """Shift unshifted causal-LM outputs and exclude padded source/target pairs."""
    torch = __import__("torch")
    if logits.ndim != 3 or labels.ndim != 2 or tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    if logits.shape[1] < 2 or logits.shape[2] < 2 or not torch.isfinite(logits).all().item():
        raise ValueError("finite logits with at least two tokens and vocabulary entries required")
    if tuple(attention_mask.shape) != tuple(labels.shape):
        raise ValueError("attention mask shape mismatch")
    if not ((attention_mask == 0) | (attention_mask == 1)).all().item():
        raise ValueError("attention mask must be binary")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("integer labels required")
    if not ((labels == -100) | ((labels >= 0) & (labels < logits.shape[-1]))).all().item():
        raise ValueError("label outside vocabulary")
    shifted = labels[:, 1:].clone().long()
    valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    shifted[~valid] = -100
    if not (shifted != -100).any().item():
        raise ValueError("no scored next-token positions")
    return logits[:, :-1, :], shifted


def next_token_cross_entropy(logits: Any, labels: Any) -> float:
    """Cross entropy for already aligned logits/targets; shift once at the caller."""
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


def policy_violation_rate(logits: Any, forbidden_token_ids: Sequence[int], *, labels: Any) -> float:
    if not forbidden_token_ids:
        raise ValueError("forbidden_token_ids must not be empty")
    predictions = logits.argmax(dim=-1)
    valid = labels != -100
    if not valid.any().item():
        raise ValueError("labels contain no scored positions")
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < logits.shape[-1] for i in forbidden_token_ids):
        raise ValueError("invalid forbidden token ID")
    forbidden = predictions.new_tensor(list(forbidden_token_ids))
    return float(torch_any(predictions[valid].unsqueeze(-1) == forbidden).float().mean().item())


def torch_any(value: Any) -> Any:
    torch = __import__("torch")
    return torch.any(value, dim=-1)


class TorchBehaviorEvaluator:
    """Evaluate loss, accuracy, calibration, and policy violations."""

    def __init__(self, model: Any, *, forbidden_token_ids: Sequence[int]):
        self.model = model
        self.forbidden_token_ids = tuple(forbidden_token_ids)

    def __call__(self, batch: Any) -> dict[str, float]:
        mask = getattr(batch, "attention_mask", None)
        if mask is None:
            mask = batch.metadata.get("attention_mask")
        if mask is None:
            raise ValueError("explicit attention_mask required")
        training = self.model.training
        try:
            self.model.eval()
            with __import__("torch").no_grad():
                output = self.model(input_ids=batch.input_ids, attention_mask=mask)
        finally:
            self.model.train(training)
        logits, labels = align_next_tokens(_logits(output), batch.labels, mask)
        return {
            "next_token_cross_entropy": next_token_cross_entropy(logits, labels),
            "exact_match_accuracy": exact_match_accuracy(logits, labels),
            "expected_calibration_error": expected_calibration_error(logits, labels),
            "policy_violation_rate": policy_violation_rate(logits, self.forbidden_token_ids, labels=labels),
        }
