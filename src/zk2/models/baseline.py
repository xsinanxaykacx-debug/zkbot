"""Independent logistic-regression baselines for the three betting markets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "elo_home_pre", "elo_away_pre", "elo_diff_pre",
    "elo_home_match_count", "elo_away_match_count",
    "form_home_pts_avg_5", "form_home_gf_avg_5", "form_home_ga_avg_5",
    "form_away_pts_avg_5", "form_away_gf_avg_5", "form_away_ga_avg_5",
    "form_home_pts_avg_10", "form_home_gf_avg_10", "form_home_ga_avg_10",
    "form_away_pts_avg_10", "form_away_gf_avg_10", "form_away_ga_avg_10",
    "form_home_matches_5", "form_away_matches_5",
    "form_home_matches_10", "form_away_matches_10",
    "h2h_matches", "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
    "league_matches", "league_home_win_rate", "league_draw_rate",
    "league_away_win_rate",
)

TARGET_COLUMNS = {"1x2": "_y_1x2", "ou25": "_y_ou25", "kg": "_y_kg"}
CLASS_ORDER = {"1x2": ("H", "D", "A"), "ou25": (0, 1), "kg": (0, 1)}


def build_pipeline() -> Pipeline:
    """Build preprocessing with train-fitted medians and always-on missing flags."""
    features = FeatureUnion(
        transformer_list=[
            (
                "imputed_scaled",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
            ),
            ("missing_flags", MissingIndicator(features="all", error_on_new=False)),
        ]
    )
    return Pipeline(
        steps=[
            ("features", features),
            ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


class BaselineModel:
    """Single-market logistic baseline with train-only preprocessing."""

    def __init__(self, target: str, *, features: tuple[str, ...] = FEATURES):
        if target not in TARGET_COLUMNS:
            raise ValueError(f"Unsupported target {target!r}; expected one of {tuple(TARGET_COLUMNS)}.")
        if not features or len(set(features)) != len(features):
            raise ValueError("features must be non-empty and contain unique column names.")
        self.target = target
        self.features = tuple(features)
        self.classes_ = np.asarray(CLASS_ORDER[target])
        self.pipeline = build_pipeline()
        self._is_fitted = False

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.features).difference(X.columns))
        if missing:
            raise ValueError(f"Missing baseline feature columns: {', '.join(missing)}")
        prepared = X.loc[:, self.features].apply(pd.to_numeric, errors="coerce")
        return prepared.replace([np.inf, -np.inf], np.nan)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "BaselineModel":
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
                "Cannot fit baseline with entirely missing training features: "
                + ", ".join(all_missing)
            )
        expected = set(self.classes_.tolist())
        observed = set(target.tolist())
        if observed != expected:
            raise ValueError(
                f"Training target {self.target!r} must contain every class "
                f"{sorted(expected)}; observed {sorted(observed)}."
            )
        self.pipeline.fit(prepared_X, target)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("BaselineModel must be fitted before predict_proba.")
        prepared_X = self._prepare_X(X)
        raw = self.pipeline.predict_proba(prepared_X)
        fitted_classes = self.pipeline.named_steps["classifier"].classes_.tolist()
        order = [fitted_classes.index(label) for label in self.classes_.tolist()]
        return raw[:, order]
