"""Leakage-aware post-hoc probability calibration."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class OneVsRestCalibrator:
    """Calibrate each class score independently, then renormalize multiclass rows."""

    def __init__(self, method: str):
        if method not in {"platt", "isotonic"}:
            raise ValueError("method must be 'platt' or 'isotonic'.")
        self.method = method
        self.models = []
        self.n_classes = 0
        self._fitted = False

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped))

    def fit(self, probabilities: np.ndarray, y_true, class_order: tuple):
        probs = np.asarray(probabilities, dtype=float)
        y = np.asarray(y_true)
        if probs.ndim != 2 or probs.shape != (len(y), len(class_order)):
            raise ValueError("probabilities shape must match y_true and class_order.")
        if len(y) < 10 or not np.isfinite(probs).all():
            raise ValueError("Calibration requires at least 10 rows and finite probabilities.")
        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Probability rows must sum to one.")
        if set(y.tolist()) != set(class_order):
            raise ValueError("Calibration fit data must contain every target class.")

        scores = self._logit(probs)
        self.models = []
        self.n_classes = len(class_order)
        for index, label in enumerate(class_order):
            binary_y = (y == label).astype(int)
            if self.method == "platt":
                model = LogisticRegression(solver="lbfgs", max_iter=1000)
                model.fit(scores[:, [index]], binary_y)
            else:
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(scores[:, index], binary_y)
            self.models.append(model)
        self._fitted = True
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator must be fitted before prediction.")
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 2 or probs.shape[1] != self.n_classes:
            raise ValueError("Input probability shape does not match fitted calibrator.")
        scores = self._logit(probs)
        calibrated = np.column_stack([
            model.predict_proba(scores[:, [i]])[:, 1]
            if self.method == "platt"
            else model.predict(scores[:, i])
            for i, model in enumerate(self.models)
        ])
        row_sums = calibrated.sum(axis=1, keepdims=True)
        invalid = ~np.isfinite(row_sums[:, 0]) | (row_sums[:, 0] <= 0)
        if invalid.any():
            calibrated[invalid] = probs[invalid]
            row_sums[invalid] = calibrated[invalid].sum(axis=1, keepdims=True)
        calibrated /= row_sums
        return calibrated
