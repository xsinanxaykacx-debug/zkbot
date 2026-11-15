import pandas as pd
import pytest

from zk2.data.dedup import auto_merge_exact, find_match_candidates


def sample(date):
    return pd.DataFrame({
        "_date": [date],
        "_team_home_canonical": ["Galatasaray"],
        "_team_away_canonical": ["Fenerbahçe"],
    })


def test_exact_match_is_separated_from_one_day_candidate():
    exact = find_match_candidates(sample("2024-01-01"), sample("2024-01-01"))
    assert len(exact["exact"]) == 1
    assert exact["tolerant"].empty

    shifted = find_match_candidates(sample("2024-01-01"), sample("2024-01-02"))
    assert shifted["exact"].empty
    assert len(shifted["tolerant"]) == 1


def test_exact_auto_merge_rejects_duplicate_keys():
    duplicated = pd.concat([sample("2024-01-01"), sample("2024-01-01")], ignore_index=True)
    with pytest.raises(ValueError):
        auto_merge_exact(duplicated, sample("2024-01-01"))
