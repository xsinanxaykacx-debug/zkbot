"""Audit Openplay's licensed match xG against the EPL.xlsm match list.

Openplay data is used under CC BY 4.0 with attribution. This workflow only
downloads the published release assets; it does not scrape FotMob.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zk2.data.adapters.epl_xlsm import EplXlsmAdapter
from zk2.data.epl_team_mapping import normalize_epl_team
from zk2.data.team_mapping import team_key

RELEASE = "data-v0.1.0"
ATTRIBUTION = "openplay (Noah Bair), CC BY 4.0"


def _normal_team(value: object) -> str:
    return team_key(normalize_epl_team(value))


def _season_from_date(value: object) -> str:
    date = pd.Timestamp(value)
    start = date.year if date.month >= 8 else date.year - 1
    return f"{start}/{start + 1}"


def prepare_openplay_matches(
    matches: pd.DataFrame, team_stats: pd.DataFrame
) -> pd.DataFrame:
    required_matches = {
        "match_id", "comp", "date", "home_team", "away_team",
        "home_score", "away_score",
    }
    required_stats = {"match_id", "comp", "side", "xg", "npxg"}
    missing_matches = sorted(required_matches.difference(matches.columns))
    missing_stats = sorted(required_stats.difference(team_stats.columns))
    if missing_matches or missing_stats:
        raise ValueError(
            f"Openplay schema missing fields: matches={missing_matches}, "
            f"team_match_stats={missing_stats}"
        )

    eng = matches.loc[matches["comp"].astype("string").str.lower().eq("eng")].copy()
    stats = team_stats.loc[
        team_stats["comp"].astype("string").str.lower().eq("eng")
    ].copy()
    stats["side"] = stats["side"].astype("string").str.lower()
    stats = stats.loc[stats["side"].isin(["home", "away"])].copy()
    if stats.duplicated(["match_id", "side"]).any():
        raise ValueError("Openplay has duplicate team-stat rows for a match/side.")

    for column in ("xg", "npxg"):
        stats[column] = pd.to_numeric(stats[column], errors="coerce")

    xg = stats.pivot(index="match_id", columns="side", values=["xg", "npxg"])
    xg.columns = [f"{metric}_{side}" for metric, side in xg.columns]
    xg = xg.reset_index()
    out = eng.merge(xg, on="match_id", how="left", validate="one_to_one")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["season"] = out["date"].map(_season_from_date)
    out["home_key"] = out["home_team"].map(_normal_team)
    out["away_key"] = out["away_team"].map(_normal_team)
    out["match_key"] = (
        out["date"].dt.strftime("%Y-%m-%d") + "|" + out["home_key"] + "|" + out["away_key"]
    )
    out["xg_complete"] = out[["xg_home", "xg_away"]].notna().all(axis=1)
    out["npxg_complete"] = out[["npxg_home", "npxg_away"]].notna().all(axis=1)
    return out


def audit_frames(
    epl: pd.DataFrame,
    matches: pd.DataFrame,
    team_stats: pd.DataFrame,
    competitions: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required_epl = {
        "_date", "_season", "_team_home_canonical", "_team_away_canonical",
        "home_score", "away_score",
    }
    missing = sorted(required_epl.difference(epl.columns))
    if missing:
        raise ValueError(f"EPL source missing required columns: {missing}")

    openplay = prepare_openplay_matches(matches, team_stats)
    source = epl.copy()
    source["_date"] = pd.to_datetime(source["_date"], errors="coerce").dt.normalize()
    source["home_key"] = source["_team_home_canonical"].map(_normal_team)
    source["away_key"] = source["_team_away_canonical"].map(_normal_team)
    source["match_key"] = (
        source["_date"].dt.strftime("%Y-%m-%d") + "|" + source["home_key"] + "|" + source["away_key"]
    )

    if source["match_key"].duplicated().any():
        raise ValueError("EPL.xlsm has duplicate date/home/away match keys.")
    if openplay["match_key"].duplicated().any():
        raise ValueError("Openplay EPL has duplicate date/home/away match keys.")

    provider_columns = [
        "match_id", "season", "home_team", "away_team", "home_score", "away_score",
        "xg_home", "xg_away", "npxg_home", "npxg_away", "xg_complete", "npxg_complete",
    ]
    joined = source.merge(
        openplay[provider_columns + ["match_key"]],
        on="match_key", how="left", validate="one_to_one", indicator=True,
        suffixes=("_epl", "_openplay"),
    )
    joined["openplay_matched"] = joined["_merge"].eq("both")
    joined["score_match"] = (
        joined["home_score_epl"].eq(joined["home_score_openplay"])
        & joined["away_score_epl"].eq(joined["away_score_openplay"])
    )
    joined["clean_xg_match"] = (
        joined["openplay_matched"] & joined["xg_complete"].fillna(False) & joined["score_match"]
    )
    joined["clean_npxg_match"] = (
        joined["openplay_matched"] & joined["npxg_complete"].fillna(False) & joined["score_match"]
    )

    season_rows = []
    for season, group in joined.groupby("_season", dropna=False, sort=True):
        n = len(group)
        season_rows.append({
            "season": str(season),
            "epl_source_matches": int(n),
            "openplay_matches_matched": int(group["openplay_matched"].sum()),
            "openplay_match_rate": float(group["openplay_matched"].mean()) if n else 0.0,
            "both_xg_and_score_verified": int(group["clean_xg_match"].sum()),
            "clean_xg_coverage": float(group["clean_xg_match"].mean()) if n else 0.0,
            "score_mismatches": int((group["openplay_matched"] & ~group["score_match"]).sum()),
        })

    competition_names = []
    for column in ("competition", "comp", "region"):
        if column in competitions.columns:
            competition_names.extend(competitions[column].dropna().astype(str).tolist())
    turkey_terms = ("turkey", "türkiye", "super lig", "süper lig")
    turkish_scope = any(
        term in value.casefold() for value in competition_names for term in turkey_terms
    )

    n = len(joined)
    summary = {
        "provider": "Openplay",
        "release": RELEASE,
        "attribution": ATTRIBUTION,
        "purpose": "Exact date + home team + away team match join against EPL.xlsm; xG only, no model training.",
        "epl_source": {
            "matches": int(n),
            "date_min": str(source["_date"].min().date()),
            "date_max": str(source["_date"].max().date()),
        },
        "openplay_epl": {
            "matches_with_events": int(len(openplay)),
            "team_match_stats_rows": int(len(team_stats.loc[
                team_stats["comp"].astype("string").str.lower().eq("eng")
            ])),
            "both_sides_xg_complete": int(openplay["xg_complete"].sum()),
            "both_sides_xg_coverage": float(openplay["xg_complete"].mean()) if len(openplay) else 0.0,
        },
        "join": {
            "matched_matches": int(joined["openplay_matched"].sum()),
            "match_rate": float(joined["openplay_matched"].mean()) if n else 0.0,
            "clean_xg_matches": int(joined["clean_xg_match"].sum()),
            "clean_xg_coverage": float(joined["clean_xg_match"].mean()) if n else 0.0,
            "score_mismatches": int((joined["openplay_matched"] & ~joined["score_match"]).sum()),
            "unmatched_source_matches": int((~joined["openplay_matched"]).sum()),
        },
        "season_summary": season_rows,
        "turkish_super_lig_in_release_scope": bool(turkish_scope),
        "missing_value_policy": "Missing xG/npxG remains null; no zero filling.",
        "model_policy": "Audit only. Do not add xG to training until exact joins and score checks are reviewed.",
    }
    joined = joined.drop(columns=["_merge"])
    return summary, joined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epl_xlsm", type=Path)
    parser.add_argument("matches_parquet", type=Path)
    parser.add_argument("team_stats_parquet", type=Path)
    parser.add_argument("competitions_parquet", type=Path)
    parser.add_argument("--json", type=Path, default=Path("audit-results/openplay_xg_coverage.json"))
    parser.add_argument("--csv", type=Path, default=Path("audit-results/openplay_epl_match_xg.csv"))
    args = parser.parse_args()

    epl = EplXlsmAdapter(args.epl_xlsm).load().transform()
    matches = pd.read_parquet(args.matches_parquet)
    team_stats = pd.read_parquet(args.team_stats_parquet)
    competitions = pd.read_parquet(args.competitions_parquet)
    summary, joined = audit_frames(epl, matches, team_stats, competitions)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.csv, index=False, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
