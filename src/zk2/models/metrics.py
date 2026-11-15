"""Probability metrics for chronological holdout evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss


def evaluate_probabilities(
    y_true,
    probabilities: np.ndarray,
    *,
    class_order: tuple[str | int, ...],
) -> dict[str, float | int]:
    """Return LogLoss and Brier with explicit class order.

    Binary Brier uses the positive-class probability. Multiclass Brier is
    the mean row-wise sum of squared one-hot errors (not class-averaged).
    """
    if len(class_order) < 2 or len(set(class_order)) != len(class_order):
        raise ValueError("class_order must contain at least two unique classes.")
    y = np.asarray(y_true)
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape != (len(y), len(class_order)):
        raise ValueError("probabilities shape must be (n_samples, n_classes).")
    if len(y) == 0:
        raise ValueError("At least one evaluation target is required.")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities must contain only finite values.")
    if ((probs < 0) | (probs > 1)).any():
        raise ValueError("probabilities must be between 0 and 1.")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to 1.")
    unknown = set(y.tolist()).difference(class_order)
    if unknown:
        raise ValueError(f"Targets contain classes outside class_order: {sorted(unknown)}")
    class_to_index = {label: index for index, label in enumerate(class_order)}
    encoded = np.zeros_like(probs)
    for row_index, label in enumerate(y.tolist()):
        encoded[row_index, class_to_index[label]] = 1.0
    loss = log_loss(y, probs, labels=list(class_order))
    if len(class_order) == 2:
        brier = float(np.mean((probs[:, 1] - encoded[:, 1]) ** 2))
    else:
        brier = float(np.mean(np.sum((probs - encoded) ** 2, axis=1)))
    return {"n": int(len(y)), "log_loss": float(loss), "brier": brier}
