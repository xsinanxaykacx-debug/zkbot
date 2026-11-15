import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_epl_elo_diff_ablation import (
    ABLATIONS, DRAW_BAND_LABELS, draw_band_summary, feature_sets,
)
from zk2.models.baseline import FEATURES


def test_controlled_elo_variants_have_expected_columns():
    sets = feature_sets()
    assert set(sets) == set(ABLATIONS)
    assert sets["full_elo_5"] == tuple(FEATURES)
    assert "elo_diff_pre" not in sets["without_elo_diff_4"]
    assert "elo_home_pre" in sets["without_elo_diff_4"]
    assert "elo_away_pre" in sets["without_elo_diff_4"]
    assert "elo_diff_pre" not in sets["ratings_only_2"]
    assert "elo_home_match_count" not in sets["ratings_only_2"]
    assert "elo_away_match_count" not in sets["ratings_only_2"]
    assert "elo_home_pre" in sets["ratings_only_2"]
    assert "elo_away_pre" in sets["ratings_only_2"]


def test_draw_bands_use_absolute_pre_match_elo_gap():
    frame = pd.DataFrame({
        "elo_diff_pre": [-250, -150, -75, -25, 25, 75, 150, 250],
        "_y_1x2": ["D", "H", "A", "D", "H", "D", "A", "H"],
    })
    p = np.tile(np.asarray([[0.3, 0.4, 0.3]]), (len(frame), 1))
    predictions = {
        "full_elo_5": p,
        "without_elo_diff_4": p,
        "ratings_only_2": p,
    }
    rows = draw_band_summary(frame, predictions)
    assert [row["absolute_elo_gap_band"] for row in rows] == list(DRAW_BAND_LABELS)
    assert sum(row["matches"] for row in rows) == len(frame)
    assert rows[0]["matches"] == 2
    assert rows[0]["actual_draw_rate"] == 0.5
    assert rows[-1]["matches"] == 2
