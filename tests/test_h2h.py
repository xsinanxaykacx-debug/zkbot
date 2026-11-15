import pandas as pd
import pytest

from zk2.features.h2h import calculate_h2h


def _frame(rows):
    return pd.DataFrame(rows, columns=[
        "match_id", "_date", "_team_home_canonical",
        "_team_away_canonical", "MSE", "MSD",
    ])


def test_h2h_starts_empty_then_tracks_previous_meeting():
    frame = _frame([
        (1, "2024-01-01", "A", "B", 2, 0),
        (2, "2024-01-08", "A", "B", 0, 1),
    ])
    result = calculate_h2h(frame)
    assert result.loc[0, "h2h_matches"] == 0
    assert pd.isna(result.loc[0, "h2h_home_win_rate"])
    assert result.loc[1, "h2h_matches"] == 1
    assert result.loc[1, "h2h_home_wins"] == 1
    assert result.loc[1, "h2h_home_win_rate"] == 1.0


def test_h2h_reverses_home_away_perspective_correctly():
    frame = _frame([
        (1, "2024-01-01", "A", "B", 2, 0),
        (2, "2024-01-08", "B", "A", 1, 0),
    ])
    result = calculate_h2h(frame)
    assert result.loc[1, "h2h_home_wins"] == 0
    assert result.loc[1, "h2h_away_wins"] == 1


def test_h2h_counts_draws():
    frame = _frame([
        (1, "2024-01-01", "A", "B", 1, 1),
        (2, "2024-01-08", "A", "B", 0, 0),
    ])
    result = calculate_h2h(frame)
    assert result.loc[1, "h2h_draws"] == 1
    assert result.loc[1, "h2h_draw_rate"] == 1.0


def test_h2h_same_day_meetings_do_not_leak():
    frame = _frame([
        (1, "2024-01-01", "A", "B", 2, 0),
        (2, "2024-01-01", "A", "B", 0, 3),
        (3, "2024-01-02", "A", "B", 1, 1),
    ])
    result = calculate_h2h(frame)
    assert result.loc[0, "h2h_matches"] == 0
    assert result.loc[1, "h2h_matches"] == 0
    assert result.loc[2, "h2h_matches"] == 2
    assert result.loc[2, "h2h_home_wins"] == 1
    assert result.loc[2, "h2h_away_wins"] == 1


def test_h2h_missing_scores_do_not_enter_history():
    frame = _frame([
        (1, "2024-01-01", "A", "B", None, None),
        (2, "2024-01-08", "A", "B", 1, 0),
    ])
    result = calculate_h2h(frame)
    assert result.loc[1, "h2h_matches"] == 0


def test_h2h_is_capped_to_latest_n_meetings():
    frame = _frame([
        (i, f"2024-01-{i:02d}", "A", "B", 1, 0)
        for i in range(1, 5)
    ])
    result = calculate_h2h(frame, max_history=2)
    assert result.loc[3, "h2h_matches"] == 2


def test_h2h_rejects_invalid_date():
    frame = _frame([(1, "bad-date", "A", "B", 1, 0)])
    with pytest.raises(ValueError, match="invalid dates"):
        calculate_h2h(frame)
