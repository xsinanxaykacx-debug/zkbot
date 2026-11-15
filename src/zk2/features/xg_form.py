"""Leakage-safe rolling expected-goals features from prior matches only."""

from __future__ import annotations

from collections import defaultdict, deque
import math

import pandas as pd


_REQUIRED = (
    "_date",
    "_team_home_canonical",
    "_team_away_canonical",
    "xg_home",
    "xg_away",
)


def calculate_xg_form(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Append pre-match rolling xG-for/against/difference averages.

    A team's history contains only prior matches with complete xG for both
    sides. Missing xG stays missing and never becomes zero. All fixtures on
    the same calendar date read the same start-of-day histories; xG results
    are appended only after every fixture on that date has been featured.
    """
    missing = sorted(set(_REQUIRED).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required xG form columns: {', '.join(missing)}")
    if not windows or any(not isinstance(w, int) or w <= 0 for w in windows):
        raise ValueError("windows must contain positive integers.")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must be unique.")
    if frame.empty:
        result = frame.copy()
        for side in ("home", "away"):
            for window in windows:
                for metric in ("for", "against", "diff", "matches"):
                    result[f"xg_form_{side}_{metric}_{'avg_' if metric != 'matches' else ''}{window}"] = pd.Series(dtype="float64")
        return result

    work = frame.copy()
    work["_xg_form_order"] = range(len(work))
    work["_xg_form_date"] = pd.to_datetime(work["_date"], errors="coerce").dt.normalize()
    if work["_xg_form_date"].isna().any():
        raise ValueError("xG form input contains missing or invalid dates.")
    if work[["_team_home_canonical", "_team_away_canonical"]].isna().any().any():
        raise ValueError("Canonical team identifiers cannot be missing.")
    if (work["_team_home_canonical"] == work["_team_away_canonical"]).any():
        raise ValueError("A team cannot play against itself in an xG form fixture.")

    for column in ("xg_home", "xg_away"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
        invalid = work[column].notna() & (
            ~work[column].map(math.isfinite) | work[column].lt(0)
        )
        if invalid.any():
            raise ValueError(f"{column} contains non-finite or negative xG values.")

    history: defaultdict[str, deque[tuple[float, float]]] = defaultdict(
        lambda: deque(maxlen=max(windows))
    )
    feature_rows: list[dict[str, float | int]] = []
    work = work.sort_values(["_xg_form_date", "_xg_form_order"], kind="stable")

    for _, group in work.groupby("_xg_form_date", sort=True):
        day_updates: list[tuple[str, tuple[float, float]]] = []
        for row in group.to_dict("records"):
            features: dict[str, float | int] = {"_xg_form_order": row["_xg_form_order"]}
            for side, team in (
                ("home", row["_team_home_canonical"]),
                ("away", row["_team_away_canonical"]),
            ):
                past = list(history[team])
                for window in windows:
                    recent = past[-window:]
                    count = len(recent)
                    values = {
                        "for": [item[0] for item in recent],
                        "against": [item[1] for item in recent],
                    }
                    for metric, samples in values.items():
                        value = sum(samples) / count if count else float("nan")
                        features[f"xg_form_{side}_{metric}_avg_{window}"] = value
                    features[f"xg_form_{side}_diff_avg_{window}"] = (
                        features[f"xg_form_{side}_for_avg_{window}"]
                        - features[f"xg_form_{side}_against_avg_{window}"]
                        if count else float("nan")
                    )
                    features[f"xg_form_{side}_matches_{window}"] = count
            feature_rows.append(features)

            home_xg, away_xg = row["xg_home"], row["xg_away"]
            if pd.isna(home_xg) or pd.isna(away_xg):
                continue
            day_updates.append(
                (row["_team_home_canonical"], (float(home_xg), float(away_xg)))
            )
            day_updates.append(
                (row["_team_away_canonical"], (float(away_xg), float(home_xg)))
            )

        for team, result in day_updates:
            history[team].append(result)

    features = pd.DataFrame(feature_rows).set_index("_xg_form_order")
    result = frame.copy()
    result["_xg_form_order"] = range(len(result))
    return result.join(features, on="_xg_form_order").drop(columns="_xg_form_order")
