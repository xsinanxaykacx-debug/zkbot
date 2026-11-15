import numpy as np
import pandas as pd
import pytest

from zk2.data.splits import chronological_split


def test_partitions_have_strict_chronological_boundaries():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    frame = pd.DataFrame({"_date": dates, "feature": range(20)})
    train, validation, test = chronological_split(frame)
    assert train["_date"].max() < validation["_date"].min()
    assert validation["_date"].max() < test["_date"].min()


@pytest.mark.parametrize(
    ("train_fraction", "validation_fraction"),
    [(0, 0.2), (1, 0.1), (0.7, 0), (0.7, 1), (0.7, 0.3)],
)
def test_rejects_invalid_split_fractions(train_fraction, validation_fraction):
    frame = pd.DataFrame({"_date": pd.date_range("2020-01-01", periods=5)})
    with pytest.raises(ValueError, match="fractions"):
        chronological_split(
            frame,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )


def test_rejects_missing_date_column():
    with pytest.raises(ValueError, match="Missing date column: _date"):
        chronological_split(pd.DataFrame({"match_id": [1, 2, 3]}))


def test_rejects_fewer_than_three_valid_distinct_dates():
    frame = pd.DataFrame({"_date": ["2020-01-01", "2020-01-02", "invalid"]})
    with pytest.raises(ValueError, match="three distinct"):
        chronological_split(frame)


def test_drops_unparseable_dates_and_preserves_rows_in_chronological_order():
    frame = pd.DataFrame({
        "_date": ["2020-01-03", "invalid", "2020-01-01", "2020-01-02"],
        "match_id": [3, 99, 1, 2],
    })

    train, validation, test = chronological_split(frame)

    combined = pd.concat([train, validation, test])
    assert combined["match_id"].tolist() == [1, 2, 3]
    assert 99 not in combined["match_id"].tolist()
    assert combined["_date"].is_monotonic_increasing


def test_split_keeps_same_date_matches_in_one_partition():
    frame = pd.DataFrame({
        "_date": pd.to_datetime(
            ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-03"]
        ),
        "match_id": np.arange(5),
    })

    train, validation, test = chronological_split(frame)

    partitions = [set(part["_date"]) for part in (train, validation, test)]
    assert all(partitions[i].isdisjoint(partitions[j]) for i in range(3) for j in range(i + 1, 3))
