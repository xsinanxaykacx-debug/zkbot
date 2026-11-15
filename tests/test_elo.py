import pandas as pd
import pytest

from zk2.features.elo import calculate_elo


def _matches(rows):
    columns = [
        "match_id", "_date", "_team_home_canonical",
        "_team_away_canonical", "MSE", "MSD",
    ]
    return pd.DataFrame(rows, columns=columns)


def test_elo_first_match_starts_at_base_rating():
    frame = _matches([(1, "2024-01-01", "GS", "FB", 2, 1)])
    result = calculate_elo(frame)
    assert result.loc[0, "elo_home_pre"] == 1500.0
    assert result.loc[0, "elo_away_pre"] == 1500.0


def test_winner_gains_and_loser_loses_rating():
    frame = _matches([
        (1, "2024-01-01", "GS", "FB", 2, 1),
        (2, "2024-01-08", "FB", "GS", 1, 1),
    ])
    result = calculate_elo(frame)
    assert result.loc[1, "elo_home_pre"] < 1500.0
    assert result.loc[1, "elo_away_pre"] > 1500.0


def test_team_rating_continues_across_league_change():
    frame = _matches([
        (1, "2024-01-01", "GS", "FB", 2, 1),
        (2, "2024-01-08", "GS", "BAS", 1, 1),
    ])
    result = calculate_elo(frame)
    assert result.loc[1, "elo_home_pre"] > 1500.0


def test_future_matches_do_not_change_past_pre_match_features():
    frame = _matches([
        (1, "2024-01-01", "A", "B", 5, 0),
        (2, "2024-01-02", "A", "C", 0, 5),
        (3, "2024-01-03", "A", "D", 0, 5),
    ])
    result = calculate_elo(frame)
    assert result.loc[1, "elo_home_pre"] > 1500.0
    assert result.loc[2, "elo_home_pre"] < result.loc[1, "elo_home_pre"]


def test_same_day_fixtures_use_one_rating_snapshot():
    frame = _matches([
        (1, "2024-01-01", "A", "B", 5, 0),
        (2, "2024-01-01", "A", "C", 0, 5),
        (3, "2024-01-02", "A", "D", 1, 1),
    ])
    result = calculate_elo(frame)
    assert result.loc[0, "elo_home_pre"] == 1500.0
    assert result.loc[1, "elo_home_pre"] == 1500.0
    # Both same-day outcomes are applied only after both fixtures are scored.
    assert result.loc[2, "elo_home_pre"] == pytest.approx(1500.0)


def test_missing_scores_do_not_update_rating_or_match_count():
    frame = _matches([
        (1, "2024-01-01", "A", "B", None, None),
        (2, "2024-01-02", "A", "C", 1, 0),
    ])
    result = calculate_elo(frame)
    assert result.loc[1, "elo_home_pre"] == 1500.0
    assert result.loc[1, "elo_home_match_count"] == 0


def test_same_day_input_order_does_not_change_next_day_rating():
    rows = [
        (1, "2024-01-01", "A", "B", 3, 0),
        (2, "2024-01-01", "A", "C", 0, 2),
        (3, "2024-01-02", "A", "D", 1, 1),
    ]
    first = calculate_elo(_matches(rows))
    second = calculate_elo(_matches([rows[1], rows[0], rows[2]]))
    assert first.loc[2, "elo_home_pre"] == pytest.approx(second.loc[2, "elo_home_pre"])


def test_invalid_dates_are_rejected():
    frame = _matches([(1, "not-a-date", "A", "B", 1, 0)])
    with pytest.raises(ValueError, match="invalid dates"):
        calculate_elo(frame)
