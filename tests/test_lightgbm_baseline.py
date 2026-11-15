import numpy as np
import pandas as pd
import pytest

from zk2.models.lightgbm_baseline import LightGBMBaseline


def _sample(n=120):
    rng = np.random.default_rng(20261115)
    x = rng.normal(size=n)
    X = pd.DataFrame({
        "elo_diff_pre": x,
        "form_home_pts_avg_5": np.where(np.arange(n) % 9 == 0, np.nan, x * 0.5),
        "h2h_home_win_rate": rng.uniform(0, 1, size=n),
    })
    y_1x2 = np.asarray(["H", "D", "A"])[np.arange(n) % 3]
    y_binary = (x > 0).astype("int8")
    return X, y_1x2, y_binary


def test_lightgbm_multiclass_returns_explicit_hda_order():
    X, y, _ = _sample()
    model = LightGBMBaseline("1x2", features=tuple(X.columns)).fit(X.iloc[:100], y[:100])
    probabilities = model.predict_proba(X.iloc[100:])
    assert probabilities.shape == (20, 3)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert model.classes_.tolist() == ["H", "D", "A"]


@pytest.mark.parametrize("market", ["ou25", "kg"])
def test_lightgbm_binary_markets_return_two_probabilities(market):
    X, _, y = _sample()
    model = LightGBMBaseline(market, features=tuple(X.columns)).fit(X.iloc[:100], y[:100])
    probabilities = model.predict_proba(X.iloc[100:])
    assert probabilities.shape == (20, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_lightgbm_rejects_missing_training_class():
    X, y, _ = _sample()
    with pytest.raises(ValueError, match="every class"):
        LightGBMBaseline("1x2", features=tuple(X.columns)).fit(X.loc[y != "D"], y[y != "D"])


def test_lightgbm_rejects_missing_feature_column():
    X, _, y = _sample()
    model = LightGBMBaseline("ou25", features=tuple(X.columns)).fit(X, y)
    with pytest.raises(ValueError, match="Missing LightGBM feature"):
        model.predict_proba(X.drop(columns=["elo_diff_pre"]))


def test_lightgbm_rejects_prediction_before_fit():
    X, _, _ = _sample()
    model = LightGBMBaseline("ou25", features=tuple(X.columns))
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict_proba(X)
