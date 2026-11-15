import pandas as pd

from zk2.data.splits import chronological_split


def test_partitions_have_strict_chronological_boundaries():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    frame = pd.DataFrame({"_date": dates, "feature": range(20)})
    train, validation, test = chronological_split(frame)
    assert train["_date"].max() < validation["_date"].min()
    assert validation["_date"].max() < test["_date"].min()
