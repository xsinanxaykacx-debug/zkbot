import numpy as np
import pandas as pd
import pytest

from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel


def _sample(n=90):
    index = np.arange(n)
    X = pd.DataFrame({
        "elo_diff_pre": np.sin(index / 5),
        "form_home_pts_avg_5": np.where(index % 7 == 0, np.nan, (index % 5) / 2),
        "h2h_home_win_rate": np.where(index % 4 == 0, np.nan, (index % 3) / 2),
    })
    y_1x2 = np.asarray(["H", "D", "A"])[index % 3]
    y_binary = (index % 2).astype("int8")
    return X, y_1x2, y_binary


def test_1x2_baseline_trains_and_returns_three_probabilities():
    X, y, _ = _sample()
    model = BaselineModel("1x2", features=tuple(X.columns)).fit(X.iloc[:75], y[:75])
    probabilities = model.predict_proba(X.iloc[75:])
    assert probabilities.shape == (15, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert model.classes_.tolist() == ["H", "D", "A"]


def test_ou25_baseline_trains_and_returns_two_probabilities():
    X, _, y = _sample()
    model = BaselineModel("ou25", features=tuple(X.columns)).fit(X.iloc[:75], y[:75])
    probabilities = model.predict_proba(X.iloc[75:])
    assert probabilities.shape == (15, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_kg_baseline_is_a_separate_binary_model():
    X, _, y = _sample()
    model = BaselineModel("kg", features=tuple(X.columns)).fit(X.iloc[:75], y[:75])
    assert model.target == "kg"
    assert model.predict_proba(X.iloc[75:]).shape == (15, 2)


def test_missing_features_are_imputed_inside_training_pipeline():
    X, y, _ = _sample()
    model = BaselineModel("1x2", features=tuple(X.columns)).fit(X.iloc[:75], y[:75])
    probabilities = model.predict_proba(X.iloc[75:])
    assert np.isfinite(probabilities).all()


def test_missing_targets_are_excluded_before_fit():
    X, y, _ = _sample()
    target = pd.Series(y, dtype="string")
    target.iloc[0] = pd.NA
    model = BaselineModel("1x2", features=tuple(X.columns)).fit(X, target)
    assert model.pipeline.named_steps["classifier"].n_iter_.size > 0


def test_fit_rejects_training_data_missing_a_class():
    X, y, _ = _sample()
    mask = y != "D"
    with pytest.raises(ValueError, match="every class"):
        BaselineModel("1x2", features=tuple(X.columns)).fit(X.loc[mask], y[mask])


def test_predict_before_fit_and_missing_feature_column_fail_clearly():
    X, _, _ = _sample()
    model = BaselineModel("ou25", features=tuple(X.columns))
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict_proba(X)
    fitted = BaselineModel("ou25", features=tuple(X.columns)).fit(X, _sample()[2])
    with pytest.raises(ValueError, match="Missing baseline feature"):
        fitted.predict_proba(X.drop(columns=["elo_diff_pre"]))


def test_chronological_split_keeps_whole_dates_in_order():
    frame = pd.DataFrame({
        "_date": pd.date_range("2014-01-01", periods=30, freq="30D").repeat(2),
        "match_id": np.arange(60),
    }).sample(frac=1, random_state=7)
    train, validation, test = chronological_split(frame)
    assert train["_date"].max() < validation["_date"].min()
    assert validation["_date"].max() < test["_date"].min()
