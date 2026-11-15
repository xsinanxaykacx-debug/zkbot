import pandas as pd

from zk2.data.targets import add_targets


def test_missing_scores_do_not_become_false_targets():
    source = pd.DataFrame({"home_score": [2, 1, None, 0], "away_score": [0, 1, 2, None]})
    result = add_targets(source)
    assert result["_y_1x2"].tolist()[:2] == ["H", "D"]
    assert pd.isna(result.loc[2, "_y_1x2"])
    assert pd.isna(result.loc[3, "_y_ou25"])
    assert pd.isna(result.loc[2, "_y_kg"])


def test_all_three_markets_are_created():
    result = add_targets(pd.DataFrame({"home_score": [2, 0], "away_score": [1, 2]}))
    assert result["_y_1x2"].tolist() == ["H", "A"]
    assert result["_y_ou25"].tolist() == [1, 0]
    assert result["_y_kg"].tolist() == [1, 0]
