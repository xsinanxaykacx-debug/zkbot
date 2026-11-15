"""Deterministic LightGBM benchmarks using the same feature contract as the LR baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from zk2.models.baseline import CLASS_ORDER, FEATURES, TARGET_COLUMNS


DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "random_state": 20261115,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


class LightGBMBaseline:
    """Single-market LightGBM model with fixed, predeclared parameters.

    Missing feature values are left as NaN for LightGBM's native missing-value
    handling. Targets are encoded using the repository's explicit class order.
    """

    def __init__(
        self,
        target: str,
        *,
        features: tuple[str, ...] = FEATURES,
        params: dict | None = None,
    ):
        if target not in TARGET_COLUMNS:
            raise ValueError(f"Unsupported target {target!r}; expected one of {tuple(TARGET_COLUMNS)}.")
        if not features or len(set(features)) != len(features):
            raise ValueError("features must be non-empty and contain unique column names.")
        self.target = target
        self.features = tuple(features)
        self.classes_ = np.asarray(CLASS_ORDER[target])
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        objective = "multiclass" if len(self.classes_) > 2 else "binary"
        self.model = LGBMClassifier(
            objective=objective,
            num_class=len(self.classes_) if objective == "multiclass" else None,
            **self.params,
        )
        self._is_fitted = False

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.features).difference(X.columns))
        if missing:
            raise ValueError(f"Missing LightGBM feature columns: {', '.join(missing)}")
        prepared = X.loc[:, self.features].apply(pd.to_numeric, errors="coerce")
        return prepared.replace([np.inf, -np.inf], np.nan)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "LightGBMBaseline":
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        prepared_X = self._prepare_X(X)
        target = pd.Series(y, index=X.index)
        valid = target.notna()
        prepared_X = prepared_X.loc[valid]
        target = target.loc[valid]
        if prepared_X.empty:
            raise ValueError("No rows with valid targets are available for training.")
        all_missing = prepared_X.columns[prepared_X.isna().all()].tolist()
        if all_missing:
            raise ValueError(
                "Cannot fit LightGBM with entirely missing training features: "
                + ", ".join(all_missing)
            )
        class_to_index = {label: index for index, label in enumerate(self.classes_.tolist())}
        observed = set(target.tolist())
        expected = set(class_to_index)
        if observed != expected:
            raise ValueError(
                f"Training target {self.target!r} must contain every class "
                f"{sorted(expected)}; observed {sorted(observed)}."
            )
        encoded = target.map(class_to_index).astype("int32")
        self.model.fit(prepared_X, encoded)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("LightGBMBaseline must be fitted before predict_proba.")
        prepared_X = self._prepare_X(X)
        probabilities = np.asarray(self.model.predict_proba(prepared_X), dtype=float)
        if probabilities.shape != (len(X), len(self.classes_)):
            raise RuntimeError("LightGBM returned probabilities with an unexpected class shape.")
        return probabilities
