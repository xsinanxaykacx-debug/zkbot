"""Cross-source match candidate generation; exact candidates are not auto-merged."""

from difflib import SequenceMatcher
from itertools import product

import pandas as pd

KEYS = ["_date", "_team_home_canonical", "_team_away_canonical"]
SCORE_PAIRS = [("home_score", "home_score"), ("away_score", "away_score"),
               ("_home_score", "_home_score"), ("_away_score", "_away_score")]


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
    """Return exact, +/-1 day and same-day fuzzy candidates for manual review."""
    if not 0 < fuzzy_threshold <= 1:
        raise ValueError("fuzzy_threshold must be in (0, 1].")
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

    # Blocking by exact date prevents a full cross-source Cartesian comparison.
    fuzzy_rows: list[dict] = []
    b_by_date = {date: group for date, group in valid_b.groupby("_date", sort=False)}
    for date, group_a in valid_a.groupby("_date", sort=False):
        group_b = b_by_date.get(date)
        if group_b is None:
            continue
        for row_a, row_b in product(group_a.to_dict("records"), group_b.to_dict("records")):
            home_sim = SequenceMatcher(
                None, str(row_a["_team_home_canonical"]), str(row_b["_team_home_canonical"])
            ).ratio()
            away_sim = SequenceMatcher(
                None, str(row_a["_team_away_canonical"]), str(row_b["_team_away_canonical"])
            ).ratio()
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


def auto_merge_exact(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    validation_confirmed: bool = False,
) -> pd.DataFrame:
    """Merge unique exact keys only after external provenance/score validation.

    Candidate discovery is safe by default. This function deliberately refuses
    to merge until the caller confirms that competition, score, odds and source
    provenance conflicts have been reviewed.
    """
    a, b = _prepare(df_a), _prepare(df_b)
    for label, frame in (("A", a), ("B", b)):
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Source {label} has duplicate exact match keys; review before merge.")

    if not validation_confirmed:
        raise ValueError(
            "Exact keys are candidates, not proof of identity. Review competition, "
            "scores, odds and source provenance, then pass validation_confirmed=True."
        )

    # Refuse known contradictory score values rather than silently combining them.
    common_score_columns = [
        (column, f"{column}_a", f"{column}_b")
        for column in ("home_score", "away_score", "_home_score", "_away_score")
        if column in a.columns and column in b.columns
    ]
    joined = a.merge(
        b, on=KEYS, how="outer", suffixes=("_a", "_b"),
        indicator=True, validate="one_to_one"
    )
    matched = joined["_merge"].eq("both")
    for original, left_col, right_col in common_score_columns:
        left, right = joined[left_col], joined[right_col]
        conflict = matched & left.notna() & right.notna() & left.ne(right)
        if conflict.any():
            raise ValueError(f"Conflicting {original} values on exact match keys; manual review required.")

    # If both inputs carry competition identity, conflicting non-null values block merge.
    for column in ("competition_id", "league_id", "_competition_canonical"):
        if column in a.columns and column in b.columns:
            left_col, right_col = f"{column}_a", f"{column}_b"
            if left_col in joined.columns and right_col in joined.columns:
                left, right = joined[left_col], joined[right_col]
                conflict = matched & left.notna() & right.notna() & left.ne(right)
                if conflict.any():
                    raise ValueError(
                        f"Conflicting {column} values on exact match keys; manual review required."
                    )
    return joined
