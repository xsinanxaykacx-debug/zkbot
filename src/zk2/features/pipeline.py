"""Unified chronological pre-match feature pipeline."""

from __future__ import annotations

import pandas as pd

from zk2.features.elo import calculate_elo
from zk2.features.form import calculate_form
from zk2.features.h2h import calculate_h2h
from zk2.features.league_strength import calculate_league_strength


def build_features(
    frame: pd.DataFrame,
    *,
    form_windows: tuple[int, ...] = (5, 10),
    h2h_max_history: int = 10,
    league_window: int = 100,
) -> pd.DataFrame:
    """Build all pre-match features while preserving the input row order.

    The input must contain canonical, league-independent team identifiers and
    a canonical league identifier. Raw scores are consumed only to update
    histories after each calendar day's feature snapshot has been created.
    """
    required = {"_date", "_team_home_canonical", "_team_away_canonical", "_league_canonical", "MSE", "MSD"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required pipeline columns: {', '.join(missing)}")

    result = calculate_elo(frame)
    result = calculate_form(result, windows=form_windows)
    result = calculate_h2h(result, max_history=h2h_max_history)
    result = calculate_league_strength(result, window=league_window)
    return result
