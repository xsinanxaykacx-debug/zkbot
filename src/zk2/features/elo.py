"""League-independent pre-match Elo features with leakage-safe daily updates."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


_REQUIRED_COLUMNS = (
    "_date",
    "_team_home_canonical",
    "_team_away_canonical",
    "MSE",
    "MSD",
)


def _k_factor(match_count: int) -> float:
    """Use faster adaptation for teams with less scored-match history."""
    if match_count < 10:
        return 40.0
    if match_count < 30:
        return 30.0
    return 20.0


def calculate_elo(
    frame: pd.DataFrame,
    *,
    base_elo: float = 1500.0,
    initial_k: float = 40.0,
    middle_k: float = 30.0,
    established_k: float = 20.0,
    first_k_matches: int = 10,
    established_after_matches: int = 30,
) -> pd.DataFrame:
    """Append pre-match Elo features without resetting teams between leagues.

    All fixtures on a calendar date receive ratings from the start of that date.
    Their rating deltas are calculated from that same snapshot and applied together
    after the date is processed. Missing scores do not update Elo or match counts.

    Team identity must already be canonical and league-independent. League strength
    and home advantage are intentionally not mixed into this pure team rating.
    """
    required = set(_REQUIRED_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required Elo columns: {', '.join(missing)}")
    if frame.empty:
        result = frame.copy()
        for column in (
            "elo_home_pre",
            "elo_away_pre",
            "elo_diff_pre",
            "elo_home_match_count",
            "elo_away_match_count",
        ):
            result[column] = pd.Series(dtype="float64" if column.startswith("elo_") and "count" not in column else "int64")
        return result
    if first_k_matches < 0 or established_after_matches < first_k_matches:
        raise ValueError("Invalid K-factor match-count thresholds.")
    if min(initial_k, middle_k, established_k) <= 0:
        raise ValueError("K factors must be positive.")

    work = frame.copy()
    work["_elo_input_order"] = range(len(work))
    work["_elo_date"] = pd.to_datetime(work["_date"], errors="coerce")
    if work["_elo_date"].isna().any():
        raise ValueError("Elo input contains missing or invalid dates.")
    if work[["_team_home_canonical", "_team_away_canonical"]].isna().any().any():
        raise ValueError("Canonical team identifiers cannot be missing.")
    if (work["_team_home_canonical"] == work["_team_away_canonical"]).any():
        raise ValueError("A team cannot play against itself in an Elo fixture.")

    ratings: defaultdict[str, float] = defaultdict(lambda: float(base_elo))
    match_counts: defaultdict[str, int] = defaultdict(int)
    feature_rows: list[dict[str, float | int]] = []

    work = work.sort_values(["_elo_date", "_elo_input_order"], kind="stable")

    for day, group in work.groupby("_elo_date", sort=True):
        # Snapshot ensures no fixture on this date can affect another fixture's
        # pre-match features or expected result.
        day_ratings = dict(ratings)
        day_counts = dict(match_counts)
        day_deltas: defaultdict[str, float] = defaultdict(float)
        day_count_increments: defaultdict[str, int] = defaultdict(int)

        for row in group.to_dict("records"):
            home = row["_team_home_canonical"]
            away = row["_team_away_canonical"]
            home_elo = day_ratings.get(home, float(base_elo))
            away_elo = day_ratings.get(away, float(base_elo))
            home_count = day_counts.get(home, 0)
            away_count = day_counts.get(away, 0)
            feature_rows.append(
                {
                    "_elo_input_order": row["_elo_input_order"],
                    "elo_home_pre": home_elo,
                    "elo_away_pre": away_elo,
                    "elo_diff_pre": home_elo - away_elo,
                    "elo_home_match_count": home_count,
                    "elo_away_match_count": away_count,
                }
            )

            home_score = pd.to_numeric(pd.Series([row["MSE"]]), errors="coerce").iloc[0]
            away_score = pd.to_numeric(pd.Series([row["MSD"]]), errors="coerce").iloc[0]
            if pd.isna(home_score) or pd.isna(away_score):
                continue

            if home_score > away_score:
                actual_home = 1.0
            elif home_score == away_score:
                actual_home = 0.5
            else:
                actual_home = 0.0

            expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))
            # One shared K preserves zero-sum rating changes even when the
            # opponents have different experience levels.
            k_home = initial_k if home_count < first_k_matches else (
                middle_k if home_count < established_after_matches else established_k
            )
            k_away = initial_k if away_count < first_k_matches else (
                middle_k if away_count < established_after_matches else established_k
            )
            k_match = (k_home + k_away) / 2.0
            delta = k_match * (actual_home - expected_home)
            day_deltas[home] += delta
            day_deltas[away] -= delta
            day_count_increments[home] += 1
            day_count_increments[away] += 1

        for team, delta in day_deltas.items():
            ratings[team] += delta
        for team, increment in day_count_increments.items():
            match_counts[team] += increment

    features = pd.DataFrame(feature_rows).set_index("_elo_input_order")
    result = frame.copy()
    result["_elo_input_order"] = range(len(result))
    result = result.join(features, on="_elo_input_order").drop(columns="_elo_input_order")
    return result
