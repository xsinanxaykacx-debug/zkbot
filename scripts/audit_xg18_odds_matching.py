#!/usr/bin/env python3
"""Read-only xg18 odds matching audit. No model training or final-test use."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CUTOFF = pd.Timestamp("2024-04-08")
LEAGUES = {
    "premier lig": "England Premier League",
    "super lig": "Türkiye Süper Lig",
    "turkiye 1 lig": "Türkiye 1. Lig",
}
TEAM_ALIASES = {
    "man utd": "manchester united", "man united": "manchester united",
    "man city": "manchester city", "spurs": "tottenham hotspur",
    "nott m forest": "nottingham forest", "wolves": "wolverhampton wanderers",
    "inter": "internazionale", "inter milan": "internazionale",
    "ac milan": "milan", "ath madrid": "atletico madrid",
    "atletico": "atletico madrid", "psg": "paris saint germain",
    "paris sg": "paris saint germain", "porto": "fc porto",
    "ajax amsterdam": "ajax", "psv eindhoven": "psv",
    "caykur rizespor": "rizespor", "ankaragucu": "mke ankaragucu",
}
ODDS = [
    "_odd_H", "_odd_D", "_odd_A", "_odd_under25", "_odd_over25",
    "_odd_kg_var", "_odd_kg_yok",
    "_odd_H_pinnacle_open", "_odd_D_pinnacle_open", "_odd_A_pinnacle_open",
    "_odd_over25_pinnacle_open", "_odd_under25_pinnacle_open",
    "_odd_H_pinnacle_close", "_odd_D_pinnacle_close", "_odd_A_pinnacle_close",
    "_odd_over25_pinnacle_close", "_odd_under25_pinnacle_close",
    "_market_average_home_1x2", "_market_average_draw_1x2",
    "_market_average_away_1x2", "_market_average_over25",
    "_market_average_under25",
]

def token(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value).strip().casefold())
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = value.replace("ı", "i")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())

def league_key(value: object) -> str:
    raw = token(value).replace("_", " ")
    return LEAGUES.get(raw, str(value).strip())

def team_key(value: object) -> str:
    raw = token(value)
    return TEAM_ALIASES.get(raw, raw)

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xg", type=Path, default=Path("xg18.csv"))
    parser.add_argument("--zk", type=Path, default=Path("zk2_locked_dataset.csv.gz"))
    parser.add_argument("--json", type=Path, default=Path("audit-results/xg18_odds_audit.json"))
    parser.add_argument("--csv", type=Path, default=Path("audit-results/xg18_matched_odds.csv"))
    args = parser.parse_args()

    # Determine eligible xG rows from dates first. Rows after the locked
    # train+validation cutoff are never used to construct match/score keys.
    date_frame = pd.read_csv(args.xg, usecols=["Date"], encoding="utf-8-sig")
    xg_dates = pd.to_datetime(date_frame["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    eligible_indices = set(xg_dates.index[xg_dates.notna() & (xg_dates <= CUTOFF)])
    xg = pd.read_csv(args.xg, encoding="utf-8-sig", low_memory=False)
    xg = xg.loc[xg.index.isin(eligible_indices)].copy()
    xg["_date"] = xg_dates.loc[xg.index].values
    xg["_league_key"] = xg["League"].map(league_key)
    xg["_home_key"] = xg["HomeTeam"].map(team_key)
    xg["_away_key"] = xg["AwayTeam"].map(team_key)
    xg["_home_score"] = pd.to_numeric(xg["FTHG"], errors="coerce")
    xg["_away_score"] = pd.to_numeric(xg["FTAG"], errors="coerce")

    zk = pd.read_csv(args.zk, compression="infer", low_memory=False)
    zk["_date"] = pd.to_datetime(zk["_date"], errors="coerce")
    zk = zk.loc[zk["_date"].notna() & (zk["_date"] <= CUTOFF)].copy()
    zk["_date_key"] = zk["_date"].dt.normalize()
    zk["_home_key"] = zk["_team_home_canonical"].map(team_key)
    zk["_away_key"] = zk["_team_away_canonical"].map(team_key)
    zk["_home_score"] = pd.to_numeric(zk["home_score"], errors="coerce")
    zk["_away_score"] = pd.to_numeric(zk["away_score"], errors="coerce")

    index = defaultdict(list)
    for i, row in zk.iterrows():
        key = (row["_date_key"], row["_league_canonical"], row["_home_key"], row["_away_key"])
        index[key].append(i)

    rows, statuses = [], Counter()
    by_league = defaultdict(Counter)
    zk_league_set = set(zk["_league_canonical"].dropna().unique())
    for _, row in xg.iterrows():
        league = str(row["League"])
        lk = row["_league_key"]
        if lk not in zk_league_set:
            candidates = []
            status = "league_not_in_zk"
        else:
            key = (row["_date"], lk, row["_home_key"], row["_away_key"])
            candidates = index.get(key, [])
            score_candidates = [
                i for i in candidates
                if pd.notna(row["_home_score"]) and pd.notna(row["_away_score"])
                and row["_home_score"] == zk.at[i, "_home_score"]
                and row["_away_score"] == zk.at[i, "_away_score"]
            ]
            if len(score_candidates) == 1:
                chosen = score_candidates[0]
                status = "matched"
            elif len(score_candidates) > 1:
                chosen = None
                status = "ambiguous_same_score"
            elif candidates:
                chosen = None
                status = "score_mismatch"
            else:
                chosen = None
                status = "no_candidate"
        if status == "matched":
            z = zk.loc[chosen]
            item = {
                "xg_row": int(row.name), "match_id": row.get("MatchID"),
                "date": str(pd.Timestamp(row["_date"]).date()), "xg_league": league,
                "zk_league": z["_league_canonical"], "home": row["HomeTeam"],
                "away": row["AwayTeam"], "home_score": int(row["_home_score"]),
                "away_score": int(row["_away_score"]), "home_xg": row["Home_xG"],
                "away_xg": row["Away_xG"], "zk_source": z["_source_dataset"],
                "zk_home": z["_team_home_canonical"], "zk_away": z["_team_away_canonical"],
                "match_status": status,
            }
            for col in ODDS:
                item[col] = z[col] if col in z.index else None
            one_x_two_groups = [
                ("_odd_H", "_odd_D", "_odd_A"),
                ("_odd_H_pinnacle_open", "_odd_D_pinnacle_open", "_odd_A_pinnacle_open"),
                ("_odd_H_pinnacle_close", "_odd_D_pinnacle_close", "_odd_A_pinnacle_close"),
                ("_market_average_home_1x2", "_market_average_draw_1x2", "_market_average_away_1x2"),
            ]
            ou_groups = [
                ("_odd_under25", "_odd_over25"),
                ("_odd_under25_pinnacle_open", "_odd_over25_pinnacle_open"),
                ("_odd_under25_pinnacle_close", "_odd_over25_pinnacle_close"),
                ("_market_average_under25", "_market_average_over25"),
            ]
            item["has_1x2"] = any(all(pd.notna(item.get(c)) for c in group) for group in one_x_two_groups)
            item["has_ou25"] = any(all(pd.notna(item.get(c)) for c in group) for group in ou_groups)
            item["has_kg"] = all(pd.notna(item.get(c)) for c in ("_odd_kg_var", "_odd_kg_yok"))
            item["odds_source"] = (
                "DATA.csv" if all(pd.notna(item.get(c)) for c in ("_odd_H", "_odd_D", "_odd_A"))
                else "EPL.xlsm/Pinnacle" if item["has_1x2"] else "none"
            )
            rows.append(item)
        else:
            rows.append({
                "xg_row": int(row.name), "match_id": row.get("MatchID"),
                "date": str(pd.Timestamp(row["_date"]).date()), "xg_league": league,
                "home": row["HomeTeam"], "away": row["AwayTeam"],
                "match_status": status, "candidate_count": len(candidates),
            })
        statuses[status] += 1
        by_league[league][status] += 1

    matched = pd.DataFrame(rows)
    matched_rows = matched.loc[matched["match_status"].eq("matched")].copy()
    market_summary = {}
    for market, col in [("1X2", "has_1x2"), ("OU25", "has_ou25"), ("KG", "has_kg")]:
        market_summary[market] = {
            "complete_rows": int(matched_rows[col].fillna(False).sum()) if col in matched_rows else 0,
            "matched_rows": int(len(matched_rows)),
            "coverage_pct": round(100 * matched_rows[col].fillna(False).mean(), 2) if len(matched_rows) and col in matched_rows else 0.0,
        }
    per_league = {}
    for league, counts in sorted(by_league.items()):
        n = sum(counts.values())
        per_league[league] = {
            "total_eligible": int(n), "matched": int(counts["matched"]),
            "statuses": dict(counts),
            "match_rate_pct": round(100 * counts["matched"] / n, 2) if n else 0.0,
        }
    zk_leagues = sorted(str(v) for v in zk["_league_canonical"].dropna().unique())
    zk_teams_by_league = {
        str(league): sorted(set(group["_team_home_canonical"].dropna().astype(str)) | set(group["_team_away_canonical"].dropna().astype(str)))
        for league, group in zk.groupby("_league_canonical")
    }
    unmatched_team_frequency = Counter()
    for _, unmatched in matched.loc[matched["match_status"].eq("no_candidate")].iterrows():
        unmatched_team_frequency[str(unmatched["home"])] += 1
        unmatched_team_frequency[str(unmatched["away"])] += 1
    xg_leagues = sorted(str(v) for v in pd.read_csv(args.xg, usecols=["League"], encoding="utf-8-sig")["League"].dropna().unique())
    summary = {
        "source_file": str(args.xg), "source_rows": int(len(date_frame)),
        "eligible_xg_rows_through_cutoff": int(len(xg)),
        "xg_rows_after_cutoff_excluded_by_date_only": int((xg_dates.notna() & (xg_dates > CUTOFF)).sum()),
        "invalid_xg_dates_excluded": int(xg_dates.isna().sum()),
        "validation_cutoff_inclusive": str(CUTOFF.date()),
        "zk_train_validation_rows": int(len(zk)),
        "zk_leagues_available": zk_leagues, "zk_teams_by_league": zk_teams_by_league,
        "xg_unmatched_team_frequency": dict(unmatched_team_frequency.most_common(30)),
        "xg_leagues": xg_leagues,
        "xg_league_mapping": {k: LEAGUES.get(token(k).replace("_", " "), "UNMAPPED") for k in xg_leagues},
        "status_counts": dict(statuses), "by_league": per_league,
        "market_odds_coverage_on_matched_rows": market_summary,
        "final_test_rows_used_for_matching": 0,
        "policy": {
            "matching": "exact date + explicit league + normalized home/away teams; full-time score confirms candidate",
            "ambiguous": "never auto-selected",
            "odds": "copied only from the exact matched ZK source row; missing odds remain null",
            "final_test": "xG rows after 2024-04-08 excluded using dates before score keys are built",
            "model_training_or_scoring": "not performed",
        },
    }
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.csv, index=False, encoding="utf-8")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
