import numpy as np
import pandas as pd
import pytest

from zk2.data.splits import chronological_split
from zk2.data.targets import add_targets
from zk2.features.pipeline import build_features
from zk2.models.baseline import BaselineModel, FEATURES
from zk2.models.metrics import evaluate_probabilities


def _synthetic_matches(n=120):
    rng = np.random.default_rng(20261115)
    teams = np.asarray(["A", "B", "C", "D", "E", "F"])
    rows = []
    for index, date in enumerate(pd.date_range("2018-01-01", periods=n, freq="7D")):
        home, away = rng.choice(teams, size=2, replace=False)
        rows.append({
            "match_id": index, "_date": date,
            "_team_home_canonical": home, "_team_away_canonical": away,
            "_league_canonical": "L1" if index % 3 else "L2",
            "MSE": int(rng.poisson(1.45)), "MSD": int(rng.poisson(1.15)),
        })
    return pd.DataFrame(rows)


def test_end_to_end_synthetic_pipeline_baseline_and_metrics():
    raw = _synthetic_matches()
    featured = build_features(raw)
    train, validation, test = chronological_split(featured)
    assert len(train) + len(validation) + len(test) == len(featured)
    assert train["_date"].max() < validation["_date"].min()
    assert validation["_date"].max() < test["_date"].min()

    targets = add_targets(featured.assign(
        home_score=featured["MSE"], away_score=featured["MSD"]
    ))
    train_targets = targets.loc[train.index]
    test_targets = targets.loc[test.index]

    for market, target_column, class_order in (
        ("1x2", "_y_1x2", ("H", "D", "A")),
        ("ou25", "_y_ou25", (0, 1)),
        ("kg", "_y_kg", (0, 1)),
    ):
        model = BaselineModel(market, features=FEATURES)
        model.fit(train_targets.loc[:, FEATURES], train_targets[target_column])
        probabilities = model.predict_proba(test_targets.loc[:, FEATURES])
        assert probabilities.shape == (len(test_targets), len(class_order))
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        valid = test_targets[target_column].notna()
        y = test_targets.loc[valid, target_column]
        if market == "1x2":
            y = y.astype("string").to_numpy()
        else:
            y = y.astype("int64").to_numpy()
        metrics = evaluate_probabilities(
            y, probabilities[valid.to_numpy()], class_order=class_order
        )
        assert metrics["n"] == int(valid.sum())
        assert np.isfinite(metrics["log_loss"])
        assert np.isfinite(metrics["brier"])


def test_pipeline_preserves_genuinely_unsorted_input_order():
    frame = _synthetic_matches(24).sample(frac=1, random_state=73)
    result = build_features(frame)
    assert result["match_id"].tolist() == frame["match_id"].tolist()
    assert result.index.tolist() == frame.index.tolist()


def test_metric_rejects_invalid_probability_rows():
    with pytest.raises(ValueError, match="sum to 1"):
        evaluate_probabilities(
            np.asarray(["H"]), np.asarray([[0.2, 0.2, 0.2]]),
            class_order=("H", "D", "A"),
        )


def test_binary_brier_uses_positive_class_probability():
    result = evaluate_probabilities(
        np.asarray([0, 1]), np.asarray([[0.8, 0.2], [0.25, 0.75]]),
        class_order=(0, 1),
    )
    assert result["brier"] == pytest.approx((0.2**2 + 0.25**2) / 2)
