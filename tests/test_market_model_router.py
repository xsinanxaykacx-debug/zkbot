import numpy as np
import pandas as pd
import pytest

from zk2.models.baseline import (
    FEATURES,
    XG_FEATURES,
    BaselineModel,
    MarketModelRouter,
)


def _routed_sample(rows_per_source=90):
    total = rows_per_source * 2
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.normal(size=(total, len(FEATURES))),
        columns=FEATURES,
    )
    frame["_source_dataset"] = (
        ["EPL.xlsm"] * rows_per_source + ["DATA.csv"] * rows_per_source
    )
    for column in XG_FEATURES:
        frame[column] = rng.normal(loc=1.2, scale=0.3, size=total)
    frame["_y_1x2"] = np.asarray(["H", "D", "A"] * ((total + 2) // 3))[:total]
    frame["_y_ou25"] = (np.arange(total) % 2).astype("int8")
    frame["_y_kg"] = ((np.arange(total) // 2) % 2).astype("int8")
    return frame


def test_ou_and_kg_route_epl_to_xg_model_and_non_epl_to_xg_free_model():
    frame = _routed_sample()
    for market in ("ou25", "kg"):
        model = MarketModelRouter(market).fit(frame)
        assert model.epl_model is not None
        assert model.non_epl_model is not None
        assert model.epl_model.features == FEATURES + XG_FEATURES
        assert model.non_epl_model.features == FEATURES
        probabilities = model.predict_proba(frame.iloc[-12:])
        assert probabilities.shape == (12, 2)
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_1x2_router_stays_global_and_xg_free():
    frame = _routed_sample()
    model = MarketModelRouter("1x2").fit(frame)
    assert model.global_model is not None
    assert model.epl_model is None
    assert model.non_epl_model is None
    assert model.global_model.features == FEATURES
    probabilities = model.predict_proba(frame.iloc[-9:])
    assert probabilities.shape == (9, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_ou_kg_router_rejects_missing_xg_feature_columns():
    frame = _routed_sample().drop(columns=[XG_FEATURES[0]])
    with pytest.raises(ValueError, match="requires rolling xG features"):
        MarketModelRouter("ou25").fit(frame)


def test_ou_kg_router_rejects_single_source_training_data():
    frame = _routed_sample()
    frame = frame.loc[frame["_source_dataset"].eq("EPL.xlsm")]
    with pytest.raises(ValueError, match="both EPL and non-EPL"):
        MarketModelRouter("kg").fit(frame)
