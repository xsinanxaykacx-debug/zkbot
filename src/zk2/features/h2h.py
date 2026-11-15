"""Leakage-safe head-to-head features, independent of league."""

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


def calculate_h2h(frame: pd.DataFrame, *, max_history: int = 10) -> pd.DataFrame:
    """Append pre-match H2H counts/rates from the latest valid meetings.

    A stored winner is a canonical team ID (or None for a draw), so statistics
    are always expressed relative to the current home/away orientation. All
    same-date fixtures read the same history snapshot and update it afterward.
    """
    missing = sorted(set(_REQUIRED).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required H2H columns: {', '.join(missing)}")
    if max_history <= 0:
        raise ValueError("max_history must be positive.")
    if frame.empty:
        result = frame.copy()
        for column in (
            "h2h_matches",
            "h2h_home_wins",
            "h2h_draws",
            "h2h_away_wins",
            "h2h_home_win_rate",
            "h2h_draw_rate",
            "h2h_away_win_rate",
        ):
            result[column] = pd.Series(dtype="float64" if "rate" in column else "int64")
        return result

    work = frame.copy()
    work["_h2h_order"] = range(len(work))
    work["_h2h_date"] = pd.to_datetime(work["_date"], errors="coerce")
    if work["_h2h_date"].isna().any():
        raise ValueError("H2H input contains missing or invalid dates.")
    if work[["_team_home_canonical", "_team_away_canonical"]].isna().any().any():
        raise ValueError("Canonical team identifiers cannot be missing.")
    if (work["_team_home_canonical"] == work["_team_away_canonical"]).any():
        raise ValueError("A team cannot play against itself in an H2H fixture.")

    history: defaultdict[tuple[str, str], deque[str | None]] = defaultdict(
        lambda: deque(maxlen=max_history)
    )
    feature_rows: list[dict[str, float | int]] = []
    work = work.sort_values(["_h2h_date", "_h2h_order"], kind="stable")

    for _, group in work.groupby("_h2h_date", sort=True):
        day_updates: list[tuple[tuple[str, str], str | None]] = []
        for row in group.to_dict("records"):
            home = row["_team_home_canonical"]
            away = row["_team_away_canonical"]
            pair = tuple(sorted((home, away)))
            past = list(history[pair])
            home_wins = sum(winner == home for winner in past)
            away_wins = sum(winner == away for winner in past)
            draws = sum(winner is None for winner in past)
            count = len(past)
            feature_rows.append(
                {
                    "_h2h_order": row["_h2h_order"],
                    "h2h_matches": count,
                    "h2h_home_wins": home_wins,
                    "h2h_draws": draws,
                    "h2h_away_wins": away_wins,
                    "h2h_home_win_rate": home_wins / count if count else float("nan"),
                    "h2h_draw_rate": draws / count if count else float("nan"),
                    "h2h_away_win_rate": away_wins / count if count else float("nan"),
                }
            )

            home_goals = pd.to_numeric(pd.Series([row["MSE"]]), errors="coerce").iloc[0]
            away_goals = pd.to_numeric(pd.Series([row["MSD"]]), errors="coerce").iloc[0]
            if pd.isna(home_goals) or pd.isna(away_goals):
                continue
            if home_goals > away_goals:
                winner = home
            elif home_goals < away_goals:
                winner = away
            else:
                winner = None
            day_updates.append((pair, winner))

        for pair, winner in day_updates:
            history[pair].append(winner)

    features = pd.DataFrame(feature_rows).set_index("_h2h_order")
    result = frame.copy()
    result["_h2h_order"] = range(len(result))
    return result.join(features, on="_h2h_order").drop(columns="_h2h_order")
