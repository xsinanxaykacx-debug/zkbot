import json

import numpy as np
import pandas as pd

from zk2.diagnostics import export_diagnostics
from zk2.models.baseline import FEATURES, TARGET_COLUMNS


def test_export_diagnostics_emits_confirmation_and_test_rows(tmp_path):
    rng = np.random.default_rng(91)
    n = 300
    frame = pd.DataFrame({"_date": pd.date_range("2020-01-01", periods=n, freq="D")})
    frame["_season"] = ["2020/2021"] * n
    frame["_source_dataset"] = ["test-source"] * n
    frame["_source_row"] = np.arange(n)
    frame["_league_canonical"] = ["league"] * n
    frame["_team_home_canonical"] = [f"home-{i % 20}" for i in range(n)]
    frame["_team_away_canonical"] = [f"away-{i % 20}" for i in range(n)]
    frame["home_score"] = np.where(np.arange(n) % 3 == 0, 2, 1)
    frame["away_score"] = np.where(np.arange(n) % 3 == 1, 2, 0)
    for feature in FEATURES:
        frame[feature] = rng.normal(size=n)
    frame["_y_1x2"] = np.tile(["H", "D", "A"], n // 3)
    frame["_y_ou25"] = np.tile([0, 1], n // 2)
    frame["_y_kg"] = np.tile([1, 0], n // 2)

    features_path = tmp_path / "features.csv"
    search_path = tmp_path / "search.json"
    csv_path = tmp_path / "diagnostics.csv"
    summary_path = tmp_path / "summary.json"
    frame.to_csv(features_path, index=False)
    search_path.write_text(json.dumps({
        "markets": {market: {"validation_winner": "logistic_regression"} for market in TARGET_COLUMNS}
    }), encoding="utf-8")

    summary = export_diagnostics(
        features_path, search_path, csv_path, summary_path, expected_test_rows=None
    )
    diagnostics = pd.read_csv(csv_path)
    assert set(diagnostics["partition"]) == {"validation_confirmation", "final_test"}
    assert diagnostics["match_id"].is_unique
    assert {"p_H", "p_D", "p_A", "p_ou25_over", "p_kg_yes"}.issubset(diagnostics.columns)
    assert {"_y_1x2", "_y_ou25", "_y_kg", "ll_1x2", "ll_ou25", "ll_kg"}.issubset(diagnostics.columns)
    assert summary["partitions"]["final_test"]["rows"] > 0
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
