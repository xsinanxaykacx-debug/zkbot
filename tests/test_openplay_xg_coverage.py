import pandas as pd
import pytest

from scripts.audit_openplay_xg_coverage import audit_frames, prepare_openplay_matches


def test_openplay_match_join_requires_exact_date_teams_and_scores():
    epl = pd.DataFrame({
        "_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "_season": ["2023/2024", "2023/2024"],
        "_team_home_canonical": ["Arsenal", "Chelsea"],
        "_team_away_canonical": ["Liverpool", "Everton"],
        "home_score": [2, 1],
        "away_score": [1, 0],
    })
    matches = pd.DataFrame({
        "match_id": [10, 11],
        "comp": ["eng", "eng"],
        "date": ["2024-01-01", "2024-01-02"],
        "home_team": ["Arsenal", "Chelsea"],
        "away_team": ["Liverpool", "Everton"],
        "home_score": [2, 1],
        "away_score": [1, 0],
    })
    stats = pd.DataFrame({
        "match_id": [10, 10, 11, 11],
        "comp": ["eng"] * 4,
        "side": ["home", "away", "home", "away"],
        "xg": [1.8, 0.7, 1.1, 0.4],
        "npxg": [1.2, 0.7, 1.1, 0.4],
    })
    competitions = pd.DataFrame({"competition": ["Premier League"], "region": ["England"]})

    summary, joined = audit_frames(epl, matches, stats, competitions)
    assert summary["join"]["matched_matches"] == 2
    assert summary["join"]["clean_xg_matches"] == 2
    assert summary["join"]["clean_xg_coverage"] == 1.0
    assert summary["turkish_super_lig_in_release_scope"] is False
    assert joined["xg_home"].tolist() == [1.8, 1.1]


def test_openplay_missing_xg_is_not_zero_filled():
    matches = pd.DataFrame({
        "match_id": [1], "comp": ["eng"], "date": ["2024-01-01"],
        "home_team": ["Arsenal"], "away_team": ["Liverpool"],
        "home_score": [1], "away_score": [0],
    })
    stats = pd.DataFrame({
        "match_id": [1, 1], "comp": ["eng", "eng"],
        "side": ["home", "away"], "xg": [None, 0.4], "npxg": [None, 0.4],
    })
    prepared = prepare_openplay_matches(matches, stats)
    assert pd.isna(prepared.loc[0, "xg_home"])
    assert prepared.loc[0, "xg_away"] == 0.4


def test_openplay_duplicate_match_keys_are_rejected():
    epl = pd.DataFrame({
        "_date": pd.to_datetime(["2024-01-01"]),
        "_season": ["2023/2024"],
        "_team_home_canonical": ["Arsenal"],
        "_team_away_canonical": ["Liverpool"],
        "home_score": [2], "away_score": [1],
    })
    matches = pd.DataFrame({
        "match_id": [1, 2], "comp": ["eng", "eng"],
        "date": ["2024-01-01", "2024-01-01"],
        "home_team": ["Arsenal", "Arsenal"], "away_team": ["Liverpool", "Liverpool"],
        "home_score": [2, 2], "away_score": [1, 1],
    })
    stats = pd.DataFrame({
        "match_id": [1, 1, 2, 2], "comp": ["eng"] * 4,
        "side": ["home", "away", "home", "away"],
        "xg": [1.0, 0.4, 1.0, 0.4], "npxg": [1.0, 0.4, 1.0, 0.4],
    })
    competitions = pd.DataFrame({"competition": ["Premier League"]})
    with pytest.raises(ValueError, match="duplicate date/home/away"):
        audit_frames(epl, matches, stats, competitions)
