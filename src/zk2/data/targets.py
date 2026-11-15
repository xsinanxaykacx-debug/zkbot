"""Leakage-safe match targets. Missing scores remain missing targets."""

import pandas as pd


def add_targets(
    frame: pd.DataFrame,
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
) -> pd.DataFrame:
    missing = [c for c in (home_score_col, away_score_col) if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")

    out = frame.copy()
    home = pd.to_numeric(out[home_score_col], errors="coerce")
    away = pd.to_numeric(out[away_score_col], errors="coerce")
    valid = home.notna() & away.notna()

    result = pd.Series(pd.NA, index=out.index, dtype="string")
    result.loc[valid & (home > away)] = "H"
    result.loc[valid & (home == away)] = "D"
    result.loc[valid & (home < away)] = "A"
    out["_y_1x2"] = result

    ou = pd.Series(pd.NA, index=out.index, dtype="Int8")
    ou.loc[valid] = ((home.loc[valid] + away.loc[valid]) > 2.5).astype("int8")
    out["_y_ou25"] = ou

    btts = pd.Series(pd.NA, index=out.index, dtype="Int8")
    btts.loc[valid] = ((home.loc[valid] > 0) & (away.loc[valid] > 0)).astype("int8")
    out["_y_kg"] = btts
    return out
