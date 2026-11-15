import numpy as np
import pandas as pd

from scripts.run_xg_ablation import (
    _date_cluster_ci,
    _match_key,
    _row_scores,
    _xg_feature_columns,
)


def test_match_key_uses_date_and_epl_canonical_team_names():
    frame = pd.DataFrame({
        "_date": ["2020-07-05"],
        "_team_home_canonical": ["Burnley"],
        "_team_away_canonical": ["Sheffield United"],
    })
    assert _match_key(frame).iloc[0] == "2020-07-05|burnley|sheffield united"


def test_xg_feature_set_contains_both_sides_and_both_windows():
    columns = _xg_feature_columns()
    assert len(columns) == 16
    assert len(set(columns)) == 16
    assert "xg_form_home_for_avg_5" in columns
    assert "xg_form_away_diff_avg_10" in columns
    assert "xg_form_home_matches_10" in columns


def test_row_scores_return_per_match_log_loss_and_brier():
    y = np.asarray(["H", "A"])
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.2, 0.6]])
    ll, brier = _row_scores(y, probabilities, ("H", "D", "A"))
    assert np.allclose(ll, [-np.log(0.8), -np.log(0.6)])
    assert np.allclose(brier, [0.06, 0.24])


def test_date_cluster_bootstrap_is_reproducible():
    dates = pd.Series(pd.to_datetime([
        "2024-01-01", "2024-01-01", "2024-01-02",
        "2024-01-03", "2024-01-03", "2024-01-04",
    ]))
    deltas = np.asarray([0.1, 0.2, -0.1, 0.3, 0.2, 0.0])
    first = _date_cluster_ci(dates, deltas, seed=17, reps=200)
    second = _date_cluster_ci(dates, deltas, seed=17, reps=200)
    assert first == second
    assert first[0] <= first[1]
