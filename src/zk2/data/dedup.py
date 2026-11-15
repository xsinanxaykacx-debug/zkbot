"""Cross-source match candidate generation; only exact keys are auto-eligible."""

from difflib import SequenceMatcher
from itertools import product

import pandas as pd

KEYS = ["_date", "_team_home_canonical", "_team_away_canonical"]


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in KEYS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing match-key columns: {missing}")
    out = frame.copy()
    out["_date"] = pd.to_datetime(out["_date"], errors="coerce").dt.normalize()
    return out


def find_match_candidates(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    fuzzy_threshold: float = 0.90,
) -> dict[str, pd.DataFrame]:
    """Return exact, +/-1 day and same-day fuzzy candidates without merging data."""
    a, b = _prepare(df_a), _prepare(df_b)
    exact = a.merge(b, on=KEYS, how="inner", suffixes=("_a", "_b"))
    exact = exact.loc[exact["_date"].notna()].copy()

    valid_a = a.loc[a["_date"].notna()].copy()
    valid_b = b.loc[b["_date"].notna()].copy()
    shifted = []
    for offset in (-1, 1):
        left = valid_a.copy()
        left["_candidate_date"] = left["_date"] + pd.Timedelta(days=offset)
        right = valid_b.rename(columns={"_date": "_candidate_date"})
        hits = left.merge(
            right,
            on=["_candidate_date", "_team_home_canonical", "_team_away_canonical"],
            how="inner",
            suffixes=("_a", "_b"),
        )
        if not hits.empty:
            hits["date_offset_days"] = offset
            shifted.append(hits)
    tolerant = pd.concat(shifted, ignore_index=True) if shifted else pd.DataFrame()

    # Fuzzy comparisons are blocked by exact date and are never auto-merged.
    fuzzy_rows: list[dict] = []
    b_by_date = {date: group for date, group in valid_b.groupby("_date", sort=False)}
    for date, group_a in valid_a.groupby("_date", sort=False):
        group_b = b_by_date.get(date)
        if group_b is None:
            continue
        for row_a, row_b in product(group_a.to_dict("records"), group_b.to_dict("records")):
            home_sim = SequenceMatcher(None, str(row_a["_team_home_canonical"]), str(row_b["_team_home_canonical"])).ratio()
            away_sim = SequenceMatcher(None, str(row_a["_team_away_canonical"]), str(row_b["_team_away_canonical"])).ratio()
            if home_sim >= fuzzy_threshold and away_sim >= fuzzy_threshold and (
                row_a["_team_home_canonical"] != row_b["_team_home_canonical"]
                or row_a["_team_away_canonical"] != row_b["_team_away_canonical"]
            ):
                fuzzy_rows.append({
                    "date": date,
                    "home_a": row_a["_team_home_canonical"],
                    "away_a": row_a["_team_away_canonical"],
                    "home_b": row_b["_team_home_canonical"],
                    "away_b": row_b["_team_away_canonical"],
                    "home_similarity": home_sim,
                    "away_similarity": away_sim,
                })
    return {"exact": exact, "tolerant": tolerant, "fuzzy": pd.DataFrame(fuzzy_rows)}


def auto_merge_exact(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Combine sources only when exact match keys are unique in both inputs."""
    a, b = _prepare(df_a), _prepare(df_b)
    for label, frame in (("A", a), ("B", b)):
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Source {label} has duplicate exact match keys; review before merge.")
    return a.merge(b, on=KEYS, how="outer", suffixes=("_a", "_b"), indicator=True, validate="one_to_one")
