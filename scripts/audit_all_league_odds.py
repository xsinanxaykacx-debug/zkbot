#!/usr/bin/env python3
"""Audit all Football-Data-style league CSVs in the repository against xg18.

Read-only. Exact date + league code + home/away teams, then FT score confirmation.
The locked final-test period is never used: xG rows after 2024-04-08 are excluded.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CUTOFF = pd.Timestamp("2024-04-08")
LEAGUE_TO_CODE = {
    "Ingiltere_League_One": "E2",
    "La_Liga": "SP1",
    "Serie_A": "I1",
    "Premier_Lig": "E0",
    "Super_Lig": "T1",
    "Belcika_Pro_League": "B1",
    "Hollanda_Eredivisie": "N1",
    "Germany_2_Bundesliga": "D2",
    "Portekiz_Liga_Portugal": "P1",
    "Bundesliga": "D1",
    "Spain_Segunda_Division": "SP2",
    "Italy_Serie_B": "I2",
    "France_Ligue_2": "F2",
    "Turkiye_1_Lig": "T2",
    "Ligue_1": "F1",
}
CODE_TO_LEAGUE = {v: k for k, v in LEAGUE_TO_CODE.items()}
TEAM_ALIASES = {
    "man utd": "manchester united", "man united": "manchester united",
    "man city": "manchester city", "nottm forest": "nottingham forest",
    "nott m forest": "nottingham forest", "wolves": "wolverhampton wanderers",
    "spurs": "tottenham hotspur", "newcastle": "newcastle united",
    "west brom": "west bromwich albion", "qpr": "queens park rangers",
    "ath madrid": "atletico madrid", "atletico": "atletico madrid",
    "ath bilbao": "athletic bilbao", "sociedad": "real sociedad",
    "betis": "real betis", "sevilla fc": "sevilla",
    "inter": "internazionale", "inter milan": "internazionale",
    "ac milan": "milan", "roma": "as roma",
    "bayern munich": "bayern munchen", "koln": "fc koln",
    "monchengladbach": "borussia monchengladbach",
    "psg": "paris saint germain", "paris sg": "paris saint germain",
    "marseille": "olympique marseille", "lyon": "olympique lyonnais",
    "caykur rizespor": "rizespor", "gaziantep": "gaziantep fk",
    "istanbul basaksehir": "basaksehir", "basaksehir fk": "basaksehir",
    "kasimpasa istanbul": "kasimpasa", "mke ankaragucu": "ankaragucu",
    "corum": "corum fk", "corum belediyespor": "corum fk",
    "st truiden": "sint truiden", "st. truiden": "sint truiden",
    "waregem": "zulte waregem", "vallecano": "rayo vallecano",
    "espanol": "espanyol", "spal": "spal 2013",
    "hertha": "hertha berlin", "bochum": "vfl bochum",
}

def norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = unicodedata.normalize("NFKD", str(value).strip().casefold())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("ı", "i")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def team(value: object) -> str:
    s = norm(value)
    return TEAM_ALIASES.get(s, s)

def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", low_memory=False)

def first_complete(row: pd.Series, groups: list[tuple[str, ...]]) -> bool:
    return any(all(c in row.index and pd.notna(row[c]) for c in group) for group in groups)

def odds_flags(row: pd.Series) -> dict[str, bool]:
    one_x_two = [
        ("AvgH", "AvgD", "AvgA"), ("MaxH", "MaxD", "MaxA"),
        ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"),
        ("B365CH", "B365CD", "B365CA"), ("AvgCH", "AvgCD", "AvgCA"),
        ("MaxCH", "MaxCD", "MaxCA"), ("PSCH", "PSCD", "PSCA"),
    ]
    ou = [
        ("Avg>2.5", "Avg<2.5"), ("Max>2.5", "Max<2.5"),
        ("B365>2.5", "B365<2.5"), ("P>2.5", "P<2.5"),
        ("AvgC>2.5", "AvgC<2.5"), ("MaxC>2.5", "MaxC<2.5"),
        ("B365C>2.5", "B365C<2.5"), ("PC>2.5", "PC<2.5"),
    ]
    btts = [("BTS",), ("BTTS",), ("B365BTTSY", "B365BTTSN"),
            ("AvgBTTSY", "AvgBTTSN"), ("B365>2.5", "B365<2.5")]
    return {
        "has_1x2": first_complete(row, one_x_two),
        "has_ou25": first_complete(row, ou),
        "has_btts": first_complete(row, btts[:-1]),
    }

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--xg", type=Path, default=Path("xg18.csv"))
    ap.add_argument("--out", type=Path, default=Path("audit-results"))
    args = ap.parse_args()

    files = sorted(p for p in Path(".").glob("*.csv")
                   if p.name.lower() not in {"xg18.csv", "xg7lig.csv"}
                   and re.match(r"^(B1|D1|D2|E0|E1|E2|E3|E4|F1|F2|I1|I2|N1|P1|SP1|SP2|T1|T2)( \(\d+\))?\.csv$", p.name, re.I))
    inventory, records = [], []
    for path in files:
        df = read_csv(path)
        if not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(df.columns):
            inventory.append({"file": path.name, "rows": len(df), "status": "unexpected_schema",
                              "columns": list(df.columns)})
            continue
        code = str(df["Div"].dropna().iloc[0]).strip() if "Div" in df and df["Div"].notna().any() else path.stem.split(" ")[0]
        dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df["_date"] = dates.dt.normalize()
        df["_home_key"] = df["HomeTeam"].map(team)
        df["_away_key"] = df["AwayTeam"].map(team)
        df["_hg"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["_ag"] = pd.to_numeric(df["FTAG"], errors="coerce")
        keys = pd.DataFrame({"date": df["_date"], "home": df["_home_key"], "away": df["_away_key"]})
        valid_keys = keys.dropna(subset=["date"])
        dup = int(valid_keys.duplicated(["date", "home", "away"], keep=False).sum())
        market = [odds_flags(row) for _, row in df.iterrows()]
        inventory.append({
            "file": path.name, "league_code": code, "league": CODE_TO_LEAGUE.get(code, "UNMAPPED"),
            "rows": int(len(df)), "date_min": str(df["_date"].min().date()) if df["_date"].notna().any() else None,
            "date_max": str(df["_date"].max().date()) if df["_date"].notna().any() else None,
            "invalid_dates": int(df["_date"].isna().sum()), "duplicate_match_key_rows": dup,
            "has_1x2_rows": int(sum(m["has_1x2"] for m in market)),
            "has_ou25_rows": int(sum(m["has_ou25"] for m in market)),
            "has_btts_rows": int(sum(m["has_btts"] for m in market)),
            "columns": list(df.columns),
        })
        for idx, row in df.iterrows():
            if pd.isna(row["_date"]):
                continue
            flags = odds_flags(row)
            records.append({
                "file": path.name, "league_code": code, "date": row["_date"],
                "home_key": row["_home_key"], "away_key": row["_away_key"],
                "home": row["HomeTeam"], "away": row["AwayTeam"],
                "home_score": row["_hg"], "away_score": row["_ag"],
                "flags": flags, "row": row,
            })

    xg = read_csv(args.xg)
    xg["_date"] = pd.to_datetime(xg["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    eligible = xg.loc[xg["_date"].notna() & (xg["_date"] <= CUTOFF)].copy()
    by_key = defaultdict(list)
    by_team_key = defaultdict(list)
    for rec in records:
        by_key[(rec["date"], rec["league_code"], rec["home_key"], rec["away_key"])].append(rec)
        by_team_key[(rec["league_code"], rec["home_key"], rec["away_key"])].append(rec)

    match_rows, unmatched_rows, statuses = [], [], Counter()
    near_date_counts = Counter()
    by_league = defaultdict(Counter)
    for idx, row in eligible.iterrows():
        code = LEAGUE_TO_CODE.get(str(row["League"]), "UNMAPPED")
        key = (row["_date"], code, team(row["HomeTeam"]), team(row["AwayTeam"]))
        candidates = by_key.get(key, [])
        scored = [r for r in candidates if pd.notna(row["FTHG"]) and pd.notna(row["FTAG"])
                  and row["FTHG"] == r["home_score"] and row["FTAG"] == r["away_score"]]
        chosen = None
        if code == "UNMAPPED":
            status = "league_unmapped"
        elif len(scored) == 1:
            chosen, status = scored[0], "matched"
        elif len(scored) > 1:
            # Multiple copies are accepted only if their odds are identical.
            sigs = {tuple((r["flags"].get(k),) + tuple(
                str(r["row"].get(c, "")) for c in ("AvgH","AvgD","AvgA","Avg>2.5","Avg<2.5")
            )) for r in scored}
            if len(sigs) == 1:
                chosen, status = scored[0], "matched_duplicate_identical"
            else:
                status = "ambiguous_conflicting_odds"
        elif candidates:
            status = "score_mismatch"
        else:
            status = "no_candidate"
        near_date = []
        if not chosen and code != "UNMAPPED":
            same_teams = by_team_key.get((code, team(row["HomeTeam"]), team(row["AwayTeam"])), [])
            for candidate in same_teams:
                delta = abs((candidate["date"] - row["_date"]).days)
                if delta <= 1 and pd.notna(row["FTHG"]) and pd.notna(row["FTAG"]) and row["FTHG"] == candidate["home_score"] and row["FTAG"] == candidate["away_score"]:
                    near_date.append((delta, candidate))
        if near_date:
            near_date_counts[str(row["League"])] += 1
        statuses[status] += 1
        by_league[str(row["League"])][status] += 1
        if not chosen:
            nearest = sorted(near_date, key=lambda x: x[0])[0] if near_date else None
            unmatched_rows.append({
                "xg_row": int(idx), "date": str(pd.Timestamp(row["_date"]).date()),
                "league": row["League"], "league_code": code,
                "home": row["HomeTeam"], "away": row["AwayTeam"],
                "home_score": row["FTHG"], "away_score": row["FTAG"],
                "status": status, "near_date_score_confirmed": bool(nearest),
                "nearest_date_delta_days": nearest[0] if nearest else None,
                "near_date_source_file": nearest[1]["file"] if nearest else None,
                "near_date_source_date": str(nearest[1]["date"].date()) if nearest else None,
                "near_date_source_home": nearest[1]["home"] if nearest else None,
                "near_date_source_away": nearest[1]["away"] if nearest else None,
            })
        if chosen:
            match_rows.append({
                "xg_row": int(idx), "match_id": row.get("MatchID"),
                "date": str(pd.Timestamp(row["_date"]).date()), "league": row["League"],
                "home": row["HomeTeam"], "away": row["AwayTeam"],
                "home_score": row["FTHG"], "away_score": row["FTAG"],
                "home_xg": row["Home_xG"], "away_xg": row["Away_xG"],
                "odds_source_file": chosen["file"], "odds_league_code": chosen["league_code"],
                "odds_home_name": chosen["home"], "odds_away_name": chosen["away"],
                "has_1x2": chosen["flags"]["has_1x2"],
                "has_ou25": chosen["flags"]["has_ou25"],
                "has_btts": chosen["flags"]["has_btts"],
                "AvgH": chosen["row"].get("AvgH"), "AvgD": chosen["row"].get("AvgD"),
                "AvgA": chosen["row"].get("AvgA"), "Avg>2.5": chosen["row"].get("Avg>2.5"),
                "Avg<2.5": chosen["row"].get("Avg<2.5"),
                "B365H": chosen["row"].get("B365H"), "B365D": chosen["row"].get("B365D"),
                "B365A": chosen["row"].get("B365A"),
                "B365>2.5": chosen["row"].get("B365>2.5"), "B365<2.5": chosen["row"].get("B365<2.5"),
                "match_status": status,
            })
    matches = pd.DataFrame(match_rows)
    matched_n = len(matches)
    market_summary = {}
    for market in ("1x2", "ou25", "btts"):
        col = "has_" + market
        market_summary[market] = {
            "rows_with_odds": int(matches[col].sum()) if matched_n else 0,
            "matched_rows": int(matched_n),
            "coverage_pct": round(100 * matches[col].mean(), 2) if matched_n else 0.0,
        }
    summary = {
        "cutoff_inclusive": str(CUTOFF.date()),
        "xg18_total_rows": int(len(xg)),
        "xg18_eligible_rows": int(len(eligible)),
        "xg18_after_cutoff_excluded": int(len(xg) - len(eligible)),
        "league_csv_files_found": int(len(files)),
        "league_csv_inventory": inventory,
        "total_odds_source_rows_with_valid_date": int(len(records)),
        "status_counts": dict(statuses),
        "near_date_score_confirmed_unmatched_by_league": dict(near_date_counts),
        "near_date_score_confirmed_unmatched_total": int(sum(near_date_counts.values())),
        "by_xg_league": {
            league: {"total": int(sum(c.values())), "matched": int(c["matched"] + c["matched_duplicate_identical"]),
                     "statuses": dict(c)}
            for league, c in sorted(by_league.items())
        },
        "market_coverage_on_matched_rows": market_summary,
        "final_test_rows_used": 0,
        "policy": "Exact date + mapped league code + normalized home/away teams; FT score confirms; conflicting duplicate odds never auto-selected; no imputation; no model training or scoring.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "league_odds_inventory.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    matches.to_csv(args.out / "xg18_all_league_odds_matches.csv", index=False, encoding="utf-8")
    pd.DataFrame(unmatched_rows).to_csv(args.out / "xg18_unmatched_league_odds_rows.csv", index=False, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
