import pandas as pd
import pytest

from zk2.features.pipeline import build_features


def _frame(rows):
    return pd.DataFrame(rows, columns=[
        "match_id", "_date", "_team_home_canonical",
        "_team_away_canonical", "_league_canonical", "MSE", "MSD",
    ])


def _feature_columns(frame):
    return [
        column for column in frame.columns
        if column.startswith(("elo_", "form_", "h2h_", "league_"))
    ]


def test_pipeline_builds_all_feature_families_and_preserves_rows():
    frame = _frame([
        (1, "2024-01-01", "A", "B", "L1", 2, 0),
        (2, "2024-01-08", "A", "C", "L1", 1, 1),
        (3, "2024-01-15", "B", "A", "L2", 0, 1),
    ])
    result = build_features(frame)
    assert result["match_id"].tolist() == [1, 2, 3]
    assert any(column.startswith("elo_") for column in result)
    assert any(column.startswith("form_") for column in result)
    assert any(column.startswith("h2h_") for column in result)
    assert any(column.startswith("league_") for column in result)
    assert result.loc[2, "elo_away_pre"] > 1500.0


@pytest.mark.leakage
def test_pipeline_future_score_changes_do_not_change_earlier_features():
    frame = _frame([
        (1, "2024-01-01", "A", "B", "L1", 2, 0),
        (2, "2024-01-08", "A", "C", "L1", 1, 1),
        (3, "2024-01-15", "A", "D", "L1", 0, 4),
        (4, "2024-01-22", "B", "C", "L2", 3, 0),
    ])
    changed = frame.copy()
    changed.loc[3, ["MSE", "MSD"]] = [0, 7]
    original_features = build_features(frame)
    changed_features = build_features(changed)
    columns = _feature_columns(original_features)
    pd.testing.assert_frame_equal(
        original_features.loc[:2, columns],
        changed_features.loc[:2, columns],
    )


@pytest.mark.leakage
def test_pipeline_same_calendar_day_different_times_do_not_leak():
    frame = _frame([
        (1, "2024-01-01 10:00", "A", "B", "L1", 5, 0),
        (2, "2024-01-01 20:00", "C", "D", "L1", 0, 5),
        (3, "2024-01-02 09:00", "A", "C", "L1", 1, 1),
    ])
    result = build_features(frame)
    assert result.loc[0, "elo_home_pre"] == 1500.0
    assert result.loc[1, "elo_home_pre"] == 1500.0
    assert pd.isna(result.loc[0, "form_home_pts_avg_5"])
    assert pd.isna(result.loc[1, "form_home_pts_avg_5"])
    assert result.loc[0, "league_matches"] == 0
    assert result.loc[1, "league_matches"] == 0
    assert result.loc[2, "league_matches"] == 2


def test_pipeline_team_elo_carries_across_leagues_but_league_history_isolated():
    frame = _frame([
        (1, "2024-01-01", "A", "B", "L1", 3, 0),
        (2, "2024-01-08", "A", "C", "L2", 1, 0),
        (3, "2024-01-15", "D", "E", "L1", 0, 2),
    ])
    result = build_features(frame)
    assert result.loc[1, "elo_home_pre"] > 1500.0
    assert result.loc[1, "league_matches"] == 0
    assert result.loc[2, "league_matches"] == 1
    assert result.loc[2, "league_away_win_rate"] == 0.0


def test_pipeline_rejects_missing_canonical_league():
    frame = _frame([(1, "2024-01-01", "A", "B", None, 1, 0)])
    with pytest.raises(ValueError, match="Canonical league"):
        build_features(frame)
