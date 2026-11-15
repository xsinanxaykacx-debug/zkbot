"""Leakage-safe rolling team form features, independent of league."""

from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd


_REQUIRED = (
    "_date",
    "_team_home_canonical",
    "_team_away_canonical",
    "MSE",
    "MSD",
)


def calculate_form(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Append pre-match rolling points/goals-for/goals-against averages.

    Each team's history is stored from that team's own perspective. All fixtures
    on the same date read the same start-of-day histories; valid results are
    appended only after every fixture on that date has been featured.
    """
    missing = sorted(set(_REQUIRED).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required form columns: {', '.join(missing)}")
    if not windows or any(not isinstance(w, int) or w <= 0 for w in windows):
        raise ValueError("windows must contain positive integers.")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must be unique.")
    if frame.empty:
        result = frame.copy()
        for side in ("home", "away"):
            for window in windows:
                for metric in ("pts", "gf", "ga", "matches"):
                    result[f"form_{side}_{metric}_avg_{window}"] = pd.Series(dtype="float64")
        return result

    work = frame.copy()
    work["_form_order"] = range(len(work))
    work["_form_date"] = pd.to_datetime(work["_date"], errors="coerce").dt.normalize()
    if work["_form_date"].isna().any():
        raise ValueError("Form input contains missing or invalid dates.")
    if work[["_team_home_canonical", "_team_away_canonical"]].isna().any().any():
        raise ValueError("Canonical team identifiers cannot be missing.")
    if (work["_team_home_canonical"] == work["_team_away_canonical"]).any():
        raise ValueError("A team cannot play against itself in a form fixture.")

    max_window = max(windows)
    history: defaultdict[str, deque[tuple[float, float, float]]] = defaultdict(
        lambda: deque(maxlen=max_window)
    )
    feature_rows: list[dict[str, float | int]] = []
    work = work.sort_values(["_form_date", "_form_order"], kind="stable")

    for _, group in work.groupby("_form_date", sort=True):
        day_updates: list[tuple[str, tuple[float, float, float]]] = []
        for row in group.to_dict("records"):
            features: dict[str, float | int] = {"_form_order": row["_form_order"]}
            for side, team in (
                ("home", row["_team_home_canonical"]),
                ("away", row["_team_away_canonical"]),
            ):
                past = list(history[team])
                for window in windows:
                    recent = past[-window:]
                    for index, metric in enumerate(("pts", "gf", "ga")):
                        value = (
                            sum(item[index] for item in recent) / len(recent)
                            if recent else float("nan")
                        )
                        features[f"form_{side}_{metric}_avg_{window}"] = value
                    features[f"form_{side}_matches_{window}"] = len(recent)
            feature_rows.append(features)

            home_goals = pd.to_numeric(pd.Series([row["MSE"]]), errors="coerce").iloc[0]
            away_goals = pd.to_numeric(pd.Series([row["MSD"]]), errors="coerce").iloc[0]
            if pd.isna(home_goals) or pd.isna(away_goals):
                continue
            if home_goals > away_goals:
                home_points, away_points = 3.0, 0.0
            elif home_goals < away_goals:
                home_points, away_points = 0.0, 3.0
            else:
                home_points = away_points = 1.0
            day_updates.append(
                (row["_team_home_canonical"], (home_points, float(home_goals), float(away_goals)))
            )
            day_updates.append(
                (row["_team_away_canonical"], (away_points, float(away_goals), float(home_goals)))
            )

        for team, result in day_updates:
            history[team].append(result)

    features = pd.DataFrame(feature_rows).set_index("_form_order")
    result = frame.copy()
    result["_form_order"] = range(len(result))
    return result.join(features, on="_form_order").drop(columns="_form_order")
