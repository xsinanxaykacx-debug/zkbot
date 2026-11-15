import json

import numpy as np
import pandas as pd

from zk2.models.baseline import FEATURES, TARGET_COLUMNS
from zk2.models.lightgbm_search import (
    SEARCH_GRID,
    paired_date_block_bootstrap,
    run_validation_search,
)


def test_search_grid_is_small_and_predeclared():
    assert 5 <= len(SEARCH_GRID) <= 10
    assert all("n_estimators" in candidate for candidate in SEARCH_GRID)


def test_paired_date_block_bootstrap_detects_direction():
    frame = pd.DataFrame({
        "_date": pd.to_datetime([
            "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02",
            "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04",
        ]),
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    lr = np.tile([0.80, 0.20], (8, 1))
    lgbm = np.tile([0.55, 0.45], (8, 1))
    result = paired_date_block_bootstrap(
        frame, "target", lr, lgbm, (0, 1), n_bootstrap=100, block_dates=2
    )
    assert result["observed_lr_minus_lightgbm_log_loss"] > 0
    assert result["ci_95_lower"] > 0
    assert result["lightgbm_supported_at_95_percent"] is True


def test_validation_search_uses_separate_selection_and_confirmation_slices(tmp_path):
    rng = np.random.default_rng(20261115)
    n = 240
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    frame = pd.DataFrame({"_date": dates})
    for feature in FEATURES:
        frame[feature] = rng.normal(size=n)
    frame["_y_1x2"] = np.tile(["H", "D", "A"], n // 3)
    frame["_y_ou25"] = np.tile([0, 1], n // 2)
    frame["_y_kg"] = np.tile([1, 0], n // 2)
    source = tmp_path / "features.csv"
    report_path = tmp_path / "report.json"
    frame.to_csv(source, index=False)

    import zk2.models.lightgbm_search as module
    original_split = module.chronological_split

    def guarded_split(data):
        train, validation, test = original_split(data)
        assert len(test) > 0
        return train, validation, test

    module.chronological_split = guarded_split
    try:
        report = run_validation_search(source, report_path)
    finally:
        module.chronological_split = original_split

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == saved
    assert report["selection_rows"] > 0
    assert report["confirmation_rows"] > 0
    for result in report["markets"].values():
        assert result["test_evaluated"] is False
        assert len(result["candidates_ranked_by_selection_log_loss"]) == len(SEARCH_GRID)
        assert result["confirmation"]["paired_block_bootstrap"]["replicates"] == 2000
        assert result["validation_winner"] in {"lightgbm", "logistic_regression"}
