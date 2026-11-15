import numpy as np
import pandas as pd

from scripts.run_epl_feature_ablation import GROUPS, _date_cluster_ci, feature_sets
from zk2.models.baseline import FEATURES


def test_feature_ablation_sets_remove_only_the_requested_group():
    sets = feature_sets()
    assert set(sets) == {"full", "no_elo", "no_form", "no_h2h", "no_league"}
    assert sets["full"] == tuple(FEATURES)
    for group, columns in GROUPS.items():
        assert set(sets[f"no_{group}"]) == set(FEATURES) - set(columns)
        assert len(sets[f"no_{group}"]) == len(FEATURES) - len(columns)


def test_date_cluster_bootstrap_is_reproducible():
    dates = pd.Series(pd.to_datetime([
        "2024-01-01", "2024-01-01", "2024-01-02",
        "2024-01-03", "2024-01-03", "2024-01-04",
    ]))
    deltas = np.asarray([0.1, 0.2, -0.1, 0.3, 0.2, 0.0])
    first = _date_cluster_ci(dates, deltas, seed=17, reps=200)
    second = _date_cluster_ci(dates, deltas, seed=17, reps=200)
    assert first == second
    assert first[0] <= first[1]
