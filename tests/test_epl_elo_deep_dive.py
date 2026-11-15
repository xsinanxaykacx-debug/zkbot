import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_epl_elo_deep_dive import ABLATIONS, ELO_FEATURES, _date_cluster_ci, feature_sets


def test_elo_ablation_contains_single_and_redundancy_aware_removals():
    sets = feature_sets()
    assert set(sets) == set(ABLATIONS)
    assert set(ABLATIONS["no_elo_diff_pre"]) == {"elo_diff_pre"}
    assert set(ABLATIONS["no_elo_home_and_away"]) == {"elo_home_pre", "elo_away_pre"}
    assert set(ABLATIONS["no_elo_match_counts"]) == {"elo_home_match_count", "elo_away_match_count"}
    assert set(ABLATIONS["no_all_elo"]) == set(ELO_FEATURES)
    for name, removed in ABLATIONS.items():
        assert set(sets[name]) == set(__import__("zk2.models.baseline", fromlist=["FEATURES"]).FEATURES) - set(removed)


def test_date_cluster_ci_is_reproducible_and_ordered():
    dates = pd.Series(pd.to_datetime([
        "2024-01-01", "2024-01-01", "2024-01-02",
        "2024-01-03", "2024-01-03", "2024-01-04",
    ]))
    deltas = np.asarray([0.1, 0.2, -0.1, 0.3, 0.2, 0.0])
    first = _date_cluster_ci(dates, deltas, seed=17, reps=200)
    assert first == _date_cluster_ci(dates, deltas, seed=17, reps=200)
    assert first[0] <= first[1]
