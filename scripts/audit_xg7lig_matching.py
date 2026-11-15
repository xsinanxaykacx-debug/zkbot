#!/usr/bin/env python3
"""Read-only xg7lig-to-ZK source matching audit.

CI execution branch marker: trigger isolated source-match audit only.

Only the chronological train + validation time range is eligible for matching.
The locked final-test partition is excluded before any score key is constructed.
No model is trained, tuned, or scored.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


LEAGUE_ALIASES = {
    "premier lig": "England Premier League",
    "premier league": "England Premier League",
    "la liga": "Spain La Liga",
    "bundesliga": "Germany Bundesliga",
    "serie a": "Italy Serie A",
    "ligue 1": "France Ligue 1",
    "super lig": "Türkiye Süper Lig",
    "süper lig": "Türkiye Süper Lig",
    "portekiz liga portugal": "Portugal Primeira Liga",
    "primeira liga": "Portugal Primeira Liga",
    "hollanda eredivisie": "Netherlands Eredivisie",
    "eredivisie": "Netherlands Eredivisie",
}

TEAM_ALIASES = {
    "man utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "nott m forest": "nottingham forest",
    "wolves": "wolverhampton wanderers",
    "west brom": "west bromwich albion",
    "wba": "west bromwich albion",
    "sheff utd": "sheffield united",
    "bournemouth": "afc bournemouth",
    "newcastle": "newcastle united",
    "brighton": "brighton hove albion",
    "inter": "internazionale",
    "inter milan": "internazionale",
    "ac milan": "milan",
    "ath madrid": "atletico madrid",
    "atletico": "atletico madrid",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "sporting lisbon": "sporting cp",
    "benfica lisbon": "benfica",
    "porto": "fc porto",
    "ajax amsterdam": "ajax",
    "psv eindhoven": "psv",
    "fenerbahce": "fenerbahce",
    "besiktas": "besiktas",
}


def token(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value).strip().casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("ı", "i")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def league_key(value: object) -> str:
    raw = token(value).replace("_", " ")
    return LEAGUE_ALIASES.get(raw, str(value).strip())


def team_key(value: object) -> str:
    raw = token(value)
    return TEAM_ALIASES.get(raw, raw)


def season_key(value: object) -> str:
    raw = str(value).strip()
    match = re.search(r"(20\d{2})\s*[/\-]\s*(20\d{2})", raw)
    return f"{match.group(1)}/{match.group(2)}" if match else raw


def numeric(value: object) -> float | None:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(result) else float(result)


def same_score(left_home: object, left_away: object, right_home: object, right_away: object) -> bool:
    lh, la, rh, ra = map(numeric, (left_home, left_away, right_home, right_away))
    return None not in (lh, la, rh, ra) and lh == rh and la == ra


def split_validation_cutoff(dates: pd.Series) -> pd.Timestamp:
    parsed = pd.to_datetime(dates, errors="coerce").dropna()
    unique_dates = pd.Index(parsed.drop_duplicates().sort_values())
    if len(unique_dates) < 3:
        raise ValueError("Need at least three unique dates to reproduce the locked chronological split.")
    train_end = max(1, min(len(unique_dates) - 2, int(len(unique_dates) * 0.70)))
    val_end = max(train_end + 1, min(len(unique_dates) - 1, int(len(unique_dates) * 0.85)))
    return pd.Timestamp(unique_dates[val_end - 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xg", type=Path, default=Path("xg7lig.csv"))
    parser.add_argument("--zk", type=Path, default=Path("zk2_locked_dataset.csv.gz"))
    parser.add_argument("--json", type=Path, default=Path("audit-results/xg7lig_match_audit.json"))
    parser.add_argument("--csv", type=Path, default=Path("audit-results/xg7lig_match_rows.csv"))
    args = parser.parse_args()

    xg = pd.read_csv(args.xg, encoding="utf-8-sig", low_memory=False)
    required_xg = {"League", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Home_xG", "Away_xG"}
    missing = sorted(required_xg.difference(xg.columns))
    if missing:
        raise ValueError(f"xg7lig.csv required fields missing: {missing}")

    # First pass reads only dates to determine the exact validation boundary.
    date_only = pd.read_csv(args.zk, compression="infer", usecols=["_date"], low_memory=False)
    validation_cutoff = split_validation_cutoff(date_only["_date"])
    del date_only

    # Second pass reads only required columns. Filter rows to train+validation
    # immediately; no final-test scores are used in keys or comparisons.
    zk_cols = [
        "_date", "_season", "_league_canonical", "_team_home_canonical",
        "_team_away_canonical", "MSE", "MSD", "_source_dataset",
    ]
    chunks = []
    for chunk in pd.read_csv(args.zk, compression="infer", usecols=zk_cols, chunksize=50000, low_memory=False):
        chunk["_date"] = pd.to_datetime(chunk["_date"], errors="coerce")
        eligible = chunk.loc[chunk["_date"].notna() & (chunk["_date"] <= validation_cutoff)].copy()
        if not eligible.empty:
            chunks.append(eligible)
    if not chunks:
        raise ValueError("No train/validation rows found before the chronological cutoff.")
    zk = pd.concat(chunks, ignore_index=True)

    # Normalize source identities and scores only after final-test rows have been excluded.
    zk["_league_key"] = zk["_league_canonical"].map(league_key)
    zk["_season_key"] = zk["_season"].map(season_key)
    zk["_home_key"] = zk["_team_home_canonical"].map(team_key)
    zk["_away_key"] = zk["_team_away_canonical"].map(team_key)
    zk["_home_score"] = pd.to_numeric(zk["MSE"], errors="coerce")
    zk["_away_score"] = pd.to_numeric(zk["MSD"], errors="coerce")
    zk["_date_key"] = zk["_date"].dt.strftime("%Y-%m-%d")

    xg["_league_key"] = xg["League"].map(league_key)
    xg["_season_key"] = xg["Season"].map(season_key)
    xg["_home_key"] = xg["HomeTeam"].map(team_key)
    xg["_away_key"] = xg["AwayTeam"].map(team_key)
    xg["_home_score"] = pd.to_numeric(xg["FTHG"], errors="coerce")
    xg["_away_score"] = pd.to_numeric(xg["FTAG"], errors="coerce")
    parsed_xg_dates = pd.to_datetime(xg["Date"], errors="coerce", format="mixed", utc=True)
    xg["_date_key"] = parsed_xg_dates.dt.strftime("%Y-%m-%d")

    by_date: dict[tuple, list[int]] = defaultdict(list)
    by_season: dict[tuple, list[int]] = defaultdict(list)
    for idx, row in zk.iterrows():
        season_key_tuple = (
            row["_league_key"], row["_season_key"], row["_home_key"], row["_away_key"]
        )
        by_season[season_key_tuple].append(idx)
        if pd.notna(row["_date_key"]) and str(row["_date_key"]).strip():
            by_date[(row["_date_key"],) + season_key_tuple[:1] + season_key_tuple[2:]].append(idx)

    output = []
    status_counts: Counter[str] = Counter()
    league_counts: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_source_counts: Counter[str] = Counter()

    for xg_idx, row in xg.iterrows():
        league = str(row["League"])
        has_date = pd.notna(row["_date_key"]) and bool(str(row["_date_key"]).strip())
        season_tuple = (row["_league_key"], row["_season_key"], row["_home_key"], row["_away_key"])
        candidates = by_date.get((row["_date_key"], row["_league_key"], row["_home_key"], row["_away_key"]), []) if has_date else by_season.get(season_tuple, [])
        candidate_source_counts["date_key" if has_date else "season_team_key"] += 1

        status = ""
        chosen = None
        score_candidates = [
            idx for idx in candidates
            if same_score(row["_home_score"], row["_away_score"], zk.at[idx, "_home_score"], zk.at[idx, "_away_score"])
        ]

        if not candidates:
            status = "no_candidate"
        elif len(candidates) == 1:
            if len(score_candidates) == 1:
                chosen = score_candidates[0]
                status = "matched_date" if has_date else "matched_season_team_score"
            else:
                status = "score_mismatch"
        elif len(score_candidates) == 1:
            chosen = score_candidates[0]
            status = "matched_score_disambiguated"
        elif len(score_candidates) > 1:
            status = "ambiguous_same_score"
        else:
            status = "score_mismatch_multiple_candidates"

        if chosen is not None:
            matched_row = zk.loc[chosen]
            output.append({
                "xg_row": int(xg_idx),
                "xg_match_id": row.get("MatchID"),
                "league": league,
                "season": row["Season"],
                "home": row["HomeTeam"],
                "away": row["AwayTeam"],
                "xg_home_score": row["_home_score"],
                "xg_away_score": row["_away_score"],
                "xg_home": numeric(row["Home_xG"]),
                "xg_away": numeric(row["Away_xG"]),
                "zk_date": matched_row["_date_key"],
                "zk_source": matched_row["_source_dataset"],
                "zk_league": matched_row["_league_canonical"],
                "zk_home": matched_row["_team_home_canonical"],
                "zk_away": matched_row["_team_away_canonical"],
                "match_status": status,
            })
        else:
            output.append({
                "xg_row": int(xg_idx),
                "xg_match_id": row.get("MatchID"),
                "league": league,
                "season": row["Season"],
                "home": row["HomeTeam"],
                "away": row["AwayTeam"],
                "xg_home_score": row["_home_score"],
                "xg_away_score": row["_away_score"],
                "candidate_count": len(candidates),
                "score_candidate_count": len(score_candidates),
                "match_status": status,
            })
        status_counts[status] += 1
        league_counts[league][status] += 1

    result = pd.DataFrame(output)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.csv, index=False, encoding="utf-8")

    zk_league_counts = zk["_league_key"].value_counts(dropna=False).to_dict()
    zk_league_season_counts = {}
    for (league_name, season_name), count in zk.groupby(["_league_key", "_season_key"], dropna=False).size().items():
        zk_league_season_counts.setdefault(str(league_name), {})[str(season_name)] = int(count)
    xg_league_mapping = xg.groupby("League")["_league_key"].first().astype(str).to_dict()
    unmatched_sample = result.loc[result["match_status"].eq("no_candidate")].head(20)
    summary = {
        "xg_rows": int(len(xg)),
        "xg_league_to_canonical": xg_league_mapping,
        "zk_league_counts_train_validation": {str(k): int(v) for k, v in zk_league_counts.items()},
        "zk_league_season_counts_train_validation": zk_league_season_counts,
        "no_candidate_sample": unmatched_sample.to_dict(orient="records"),
        "zk_train_validation_rows": int(len(zk)),
        "validation_cutoff_inclusive": str(validation_cutoff.date()),
        "final_test_rows_used_for_matching": 0,
        "match_methods": dict(candidate_source_counts),
        "status_counts": dict(status_counts),
        "by_league": {
            league: {
                "total": int(sum(counts.values())),
                "statuses": dict(counts),
                "matched": int(sum(v for k, v in counts.items() if k.startswith("matched_"))),
                "match_rate_pct": round(
                    100 * sum(v for k, v in counts.items() if k.startswith("matched_")) / sum(counts.values()), 2
                ),
            }
            for league, counts in sorted(league_counts.items())
        },
        "policy": {
            "method": "date+league+home+away where xG date exists; otherwise league+season+home+away then full-time score confirmation",
            "unresolved_multiple_score_matches": "ambiguous; never auto-selected",
            "final_test": "excluded before score-key construction; no labels/metrics read for matching",
            "training_or_model_evaluation": "not performed",
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
