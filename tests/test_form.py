import pandas as pd
import pytest

from zk2.features.form import calculate_form


def _frame(rows):
    return pd.DataFrame(rows, columns=[
        "match_id", "_date", "_team_home_canonical",
        "_team_away_canonical", "MSE", "MSD",
    ])


def test_form_uses_only_previous_matches():
    frame = _frame([
        (1, "2024-01-01", "A", "B", 2, 0),
        (2, "2024-01-02", "A", "C", 0, 3),
    ])
    result = calculate_form(frame)
    assert pd.isna(result.loc[0, "form_home_pts_avg_5"])
    assert result.loc[1, "form_home_pts_avg_5"] == 3.0
    assert result.loc[1, "form_home_gf_avg_5"] == 2.0
    assert result.loc[1, "form_home_ga_avg_5"] == 0.0


def test_form_is_team_perspective_when_venue_changes():
    frame = _frame([
        (1, "2024-01-01", "A", "B", 0, 2),
        (2, "2024-01-08", "C", "A", 1, 1),
    ])
    result = calculate_form(frame)
    assert result.loc[1, "form_away_pts_avg_5"] == 0.0
    assert result.loc[1, "form_away_gf_avg_5"] == 0.0
    assert result.loc[1, "form_away_ga_avg_5"] == 2.0


def test_same_day_form_updates_are_deferred():
    frame = _frame([
        (1, "2024-01-01", "A", "B", 4, 0),
        (2, "2024-01-01", "A", "C", 0, 4),
        (3, "2024-01-02", "A", "D", 1, 1),
    ])
    result = calculate_form(frame)
    assert pd.isna(result.loc[0, "form_home_pts_avg_5"])
    assert pd.isna(result.loc[1, "form_home_pts_avg_5"])
    assert result.loc[2, "form_home_pts_avg_5"] == 1.5


def test_missing_scores_do_not_enter_form_history():
    frame = _frame([
        (1, "2024-01-01", "A", "B", None, None),
        (2, "2024-01-02", "A", "C", 1, 0),
    ])
    result = calculate_form(frame)
    assert pd.isna(result.loc[1, "form_home_pts_avg_5"])
    assert result.loc[1, "form_home_matches_5"] == 0


def test_form_windows_are_capped_independently():
    frame = _frame([
        (i, f"2024-01-{i:02d}", "A", f"T{i}", 1, 0)
        for i in range(1, 8)
    ])
    result = calculate_form(frame)
    assert result.loc[6, "form_home_matches_5"] == 5
    assert result.loc[6, "form_home_matches_10"] == 6


def test_form_rejects_invalid_date():
    frame = _frame([(1, "bad-date", "A", "B", 1, 0)])
    with pytest.raises(ValueError, match="invalid dates"):
        calculate_form(frame)


def test_same_calendar_day_different_times_do_not_leak():
    frame = _frame([
        (1, "2024-01-01 10:00", "A", "B", 3, 0),
        (2, "2024-01-01 20:00", "A", "C", 0, 2),
        (3, "2024-01-02 09:00", "A", "D", 1, 1),
    ])
    result = calculate_form(frame)
    assert pd.isna(result.loc[0, "form_home_pts_avg_5"])
    assert pd.isna(result.loc[1, "form_home_pts_avg_5"])
    assert result.loc[2, "form_home_matches_5"] == 2
