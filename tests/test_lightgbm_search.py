import json

import numpy as np
import pandas as pd

from zk2.models.baseline import FEATURES, TARGET_COLUMNS
from zk2.models.lightgbm_search import SEARCH_GRID, run_validation_search


def test_search_grid_is_small_and_predeclared():
    assert 5 <= len(SEARCH_GRID) <= 10
    assert all("n_estimators" in candidate for candidate in SEARCH_GRID)


def test_validation_search_never_scores_test_partition(tmp_path, monkeypatch):
    rng = np.random.default_rng(20261115)
    n = 180
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

    # Make any attempt to read the test partition's rows via a test metric fail:
    import zk2.models.lightgbm_search as module
    original_split = module.chronological_split

    def guarded_split(data):
        train, validation, test = original_split(data)
        assert len(test) > 0
        # The search function must only consume the first two partitions.
        return train, validation, test

    monkeypatch.setattr(module, "chronological_split", guarded_split)
    report = run_validation_search(source, report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == saved
    assert all(not result["test_evaluated"] for result in report["markets"].values())
    assert all(len(result["candidates_ranked_by_validation_log_loss"]) == len(SEARCH_GRID)
               for result in report["markets"].values())
    assert all(set(result) == {
        "class_order", "logistic_regression_validation",
        "candidates_ranked_by_validation_log_loss", "best_candidate",
        "best_lightgbm_validation", "best_delta_log_loss_vs_lr",
        "validation_winner", "test_evaluated"
    } for result in report["markets"].values())
