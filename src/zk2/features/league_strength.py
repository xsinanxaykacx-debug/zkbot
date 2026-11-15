"""Leakage-safe rolling league-result context features."""

from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd


_REQUIRED = ("_date", "_league_canonical", "MSE", "MSD")


def calculate_league_strength(
    frame: pd.DataFrame,
    *,
    window: int = 100,
) -> pd.DataFrame:
    """Append pre-match league outcome rates from the latest valid results.

    Each league has an independent history. Fixtures on the same calendar day
    read the same start-of-day snapshot; results are added after all fixtures
    for that date have been featured. Missing scores are ignored.
    """
    missing = sorted(set(_REQUIRED).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required league-strength columns: {', '.join(missing)}")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")

    result = frame.copy()
    if frame.empty:
        for column in (
            "league_matches", "league_home_win_rate",
            "league_draw_rate", "league_away_win_rate",
        ):
            result[column] = pd.Series(dtype="float64" if "rate" in column else "int64")
        return result

    work = frame.copy()
    work["_league_order"] = range(len(work))
    work["_league_date"] = pd.to_datetime(work["_date"], errors="coerce").dt.normalize()
    if work["_league_date"].isna().any():
        raise ValueError("League-strength input contains missing or invalid dates.")
    if work["_league_canonical"].isna().any():
        raise ValueError("Canonical league identifiers cannot be missing.")

    history: defaultdict[str, deque[str]] = defaultdict(lambda: deque(maxlen=window))
    feature_rows: list[dict[str, float | int]] = []
    work = work.sort_values(["_league_date", "_league_order"], kind="stable")

    for _, group in work.groupby("_league_date", sort=True):
        day_updates: list[tuple[str, str]] = []
        for row in group.to_dict("records"):
            league = row["_league_canonical"]
            past = list(history[league])
            total = len(past)
            home_wins = sum(outcome == "H" for outcome in past)
            draws = sum(outcome == "D" for outcome in past)
            away_wins = sum(outcome == "A" for outcome in past)
            feature_rows.append({
                "_league_order": row["_league_order"],
                "league_matches": total,
                "league_home_win_rate": home_wins / total if total else float("nan"),
                "league_draw_rate": draws / total if total else float("nan"),
                "league_away_win_rate": away_wins / total if total else float("nan"),
            })

            home_goals = pd.to_numeric(pd.Series([row["MSE"]]), errors="coerce").iloc[0]
            away_goals = pd.to_numeric(pd.Series([row["MSD"]]), errors="coerce").iloc[0]
            if pd.isna(home_goals) or pd.isna(away_goals):
                continue
            outcome = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
            day_updates.append((league, outcome))

        for league, outcome in day_updates:
            history[league].append(outcome)

    features = pd.DataFrame(feature_rows).set_index("_league_order")
    result["_league_order"] = range(len(result))
    return result.join(features, on="_league_order").drop(columns="_league_order")
