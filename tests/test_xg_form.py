import pandas as pd
import pytest

from zk2.features.xg_form import calculate_xg_form


def _frame(rows):
    return pd.DataFrame(rows, columns=[
        "match_id", "_date", "_team_home_canonical",
        "_team_away_canonical", "xg_home", "xg_away",
    ])


@pytest.mark.leakage
def test_xg_form_uses_only_previous_matches():
    frame = _frame([
        (1, "2024-01-01", "A", "B", 2.0, 0.5),
        (2, "2024-01-08", "A", "C", 1.0, 1.0),
    ])
    result = calculate_xg_form(frame)
    assert pd.isna(result.loc[0, "xg_form_home_for_avg_5"])
    assert result.loc[1, "xg_form_home_for_avg_5"] == 2.0
    assert result.loc[1, "xg_form_home_against_avg_5"] == 0.5
    assert result.loc[1, "xg_form_home_diff_avg_5"] == 1.5
    assert result.loc[1, "xg_form_home_matches_5"] == 1


@pytest.mark.leakage
def test_same_day_xg_updates_are_deferred():
    frame = _frame([
        (1, "2024-01-01 10:00", "A", "B", 4.0, 0.0),
        (2, "2024-01-01 20:00", "A", "C", 0.0, 3.0),
        (3, "2024-01-02 09:00", "A", "D", 2.0, 1.0),
    ])
    result = calculate_xg_form(frame)
    assert pd.isna(result.loc[0, "xg_form_home_for_avg_5"])
    assert pd.isna(result.loc[1, "xg_form_home_for_avg_5"])
    assert result.loc[2, "xg_form_home_for_avg_5"] == 2.0
    assert result.loc[2, "xg_form_home_against_avg_5"] == 1.5
    assert result.loc[2, "xg_form_home_matches_5"] == 2


def test_missing_xg_is_not_zero_filled_or_added_to_history():
    frame = _frame([
        (1, "2024-01-01", "A", "B", None, None),
        (2, "2024-01-08", "A", "C", 1.2, 0.8),
    ])
    result = calculate_xg_form(frame)
    assert pd.isna(result.loc[1, "xg_form_home_for_avg_5"])
    assert pd.isna(result.loc[1, "xg_form_home_against_avg_5"])
    assert result.loc[1, "xg_form_home_matches_5"] == 0


def test_xg_form_windows_are_capped_independently():
    frame = _frame([
        (i, f"2024-01-{i:02d}", "A", f"T{i}", float(i), 0.5)
        for i in range(1, 8)
    ])
    result = calculate_xg_form(frame)
    assert result.loc[6, "xg_form_home_matches_5"] == 5
    assert result.loc[6, "xg_form_home_matches_10"] == 6
    assert result.loc[6, "xg_form_home_for_avg_5"] == 5.0


def test_xg_form_rejects_negative_or_non_finite_xg():
    for invalid in (-0.1, float("inf")):
        frame = _frame([(1, "2024-01-01", "A", "B", invalid, 0.5)])
        with pytest.raises(ValueError, match="non-finite or negative"):
            calculate_xg_form(frame)


def test_xg_form_rejects_invalid_date_and_duplicate_team():
    with pytest.raises(ValueError, match="invalid dates"):
        calculate_xg_form(_frame([(1, "bad-date", "A", "B", 1.0, 0.5)]))
    with pytest.raises(ValueError, match="against itself"):
        calculate_xg_form(_frame([(1, "2024-01-01", "A", "A", 1.0, 0.5)]))
