#!/usr/bin/env python3
"""Read-only structural and coverage audit for xg7lig.csv.

This script audits the source file only. It does not join it to model partitions,
train models, calculate prediction metrics, or read the locked final holdout.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ALIASES = {
    "league": ["league", "lig", "competition", "tournament", "country_league"],
    "match_id": ["matchid", "match_id", "fixtureid", "gameid"],
    "season": ["season", "sezon", "season_name", "seasonid"],
    "date": ["date", "matchdate", "fixturedate", "tarih", "game_date"],
    "home": ["hometeam", "home", "home_team", "ev", "evsahibi", "hometeamname"],
    "away": ["awayteam", "away", "away_team", "deplasman", "awayteamname"],
    "home_xg": ["homexg", "xghome", "home_xg", "xg_h", "xg1", "hxg"],
    "away_xg": ["awayxg", "xgaway", "away_xg", "xg_a", "xg2", "axg"],
}


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(columns: list[str], aliases: list[str]) -> str | None:
    lookup = {norm(c): c for c in columns}
    for alias in aliases:
        if norm(alias) in lookup:
            return lookup[norm(alias)]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="xg7lig.csv")
    parser.add_argument("--json", default="audit-results/xg7lig_audit.json")
    args = parser.parse_args()

    path = Path(args.csv_path)
    if not path.is_file():
        raise SystemExit(f"CSV bulunamadı: {path}")

    # Sniff delimiter; tolerate UTF-8 BOM and common Windows exports.
    df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
    cols = [str(c) for c in df.columns]
    found = {key: find_column(cols, aliases) for key, aliases in ALIASES.items()}

    print("=== xg7lig.csv GENEL BİLGİ ===")
    print(f"Dosya: {path}")
    print(f"Toplam maç satırı: {len(df):,}")
    print(f"Toplam sütun: {len(cols)}")
    print(f"Sütunlar: {cols}")
    print(f"Algılanan alanlar: {json.dumps(found, ensure_ascii=False)}")

    report: dict[str, object] = {
        "file": str(path),
        "rows": int(len(df)),
        "column_count": len(cols),
        "columns": cols,
        "detected_columns": found,
    }

    league_col = found["league"]
    season_col = found["season"]
    date_col = found["date"]
    home_col = found["home"]
    away_col = found["away"]

    if league_col:
        league_counts = df[league_col].fillna("<BOŞ>").astype(str).value_counts(dropna=False)
        print("\n=== LİG DAĞILIMI ===")
        print(league_counts.to_string())
        report["league_counts"] = {str(k): int(v) for k, v in league_counts.items()}
        report["league_count"] = int(df[league_col].nunique(dropna=True))
        print(f"Toplam farklı lig: {report['league_count']}")
    else:
        print("\nUYARI: Lig sütunu otomatik tanınamadı; lig dağılımı üretilemedi.")

    if league_col and season_col:
        counts = (df.groupby([league_col, season_col], dropna=False)
                    .size().unstack(fill_value=0))
        print("\n=== LİG × SEZON MAÇ SAYISI ===")
        print(counts.to_string())
        report["league_season_counts"] = {
            str(league): {str(season): int(n) for season, n in row.items()}
            for league, row in counts.iterrows()
        }
        print(f"Toplam sezon etiketi: {df[season_col].nunique(dropna=True)}")
    else:
        print("\nUYARI: Lig veya sezon sütunu bulunamadı; çapraz tablo üretilemedi.")

    parsed_dates = None
    if date_col:
        parsed_dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed", utc=True)
        valid = parsed_dates.dropna()
        missing_dates = int(parsed_dates.isna().sum())
        print("\\n=== TARİH KAPSAMI ===")
        print(f"İlk tarih: {valid.min() if not valid.empty else 'YOK'}")
        print(f"Son tarih: {valid.max() if not valid.empty else 'YOK'}")
        print(f"Geçerli tarih: {int(parsed_dates.notna().sum())}")
        print(f"Geçersiz/boş tarih: {missing_dates}")
        report["date_min"] = str(valid.min()) if not valid.empty else None
        report["date_max"] = str(valid.max()) if not valid.empty else None
        report["valid_dates"] = int(parsed_dates.notna().sum())
        report["invalid_dates"] = missing_dates
        if league_col:
            date_coverage = pd.DataFrame({
                "league": df[league_col].fillna("<BOŞ>").astype(str),
                "date_valid": parsed_dates.notna(),
            }).groupby("league")["date_valid"].agg(["sum", "count"])
            date_coverage["missing"] = date_coverage["count"] - date_coverage["sum"]
            date_coverage["coverage_pct"] = (date_coverage["sum"] / date_coverage["count"] * 100).round(2)
            print("\\n=== LİG BAZINDA TARİH DOLULUĞU ===")
            print(date_coverage.to_string())
            report["date_coverage_by_league"] = {
                str(k): {str(col): (float(v) if col == "coverage_pct" else int(v))
                         for col, v in row.items()}
                for k, row in date_coverage.iterrows()
            }
        if season_col:
            season_coverage = pd.DataFrame({
                "season": df[season_col].fillna("<BOŞ>").astype(str),
                "date_valid": parsed_dates.notna(),
            }).groupby("season")["date_valid"].agg(["sum", "count"])
            season_coverage["missing"] = season_coverage["count"] - season_coverage["sum"]
            season_coverage["coverage_pct"] = (season_coverage["sum"] / season_coverage["count"] * 100).round(2)
            print("\\n=== SEZON BAZINDA TARİH DOLULUĞU ===")
            print(season_coverage.to_string())
            report["date_coverage_by_season"] = {
                str(k): {str(col): (float(v) if col == "coverage_pct" else int(v))
                         for col, v in row.items()}
                for k, row in season_coverage.iterrows()
            }
    else:
        print("\\nUYARI: Tarih sütunu otomatik tanınamadı.")

    xg_cols = [c for c in cols if "xg" in norm(c)]
    xg_report: dict[str, object] = {}
    print("\n=== xG DOLULUK ORANI ===")
    if not xg_cols:
        print("xG adında sütun bulunamadı.")
    for col in xg_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        overall = float(numeric.notna().mean() * 100) if len(df) else 0.0
        print(f"{col}: genel {overall:.2f}% ({int(numeric.notna().sum())}/{len(df)})")
        by_league: dict[str, float] = {}
        if league_col:
            by_league = {
                str(k): round(float(v * 100), 2)
                for k, v in numeric.notna().groupby(df[league_col].fillna("<BOŞ>")).mean().items()
            }
            print(pd.Series(by_league, name=f"{col} doluluk %").to_string())
        xg_report[col] = {"overall_pct": round(overall, 4), "by_league_pct": by_league}
    report["xg_completeness"] = xg_report

    print("\n=== TEKRAR EDEN MAÇ KONTROLÜ ===")
    duplicate_report: dict[str, object] = {}
    if found["match_id"]:
        ids = df[found["match_id"]].astype("string").str.strip()
        id_dups = int(ids.duplicated(keep="first").sum())
        duplicate_report["duplicate_match_id_rows"] = id_dups
        print(f"Tekrar eden MatchID satırı: {id_dups}")
    if date_col and home_col and away_col and parsed_dates is not None:
        valid_mask = parsed_dates.notna()
        key = pd.DataFrame({
            "date": parsed_dates.loc[valid_mask].dt.strftime("%Y-%m-%d"),
            "home": df.loc[valid_mask, home_col].astype("string").str.strip().str.casefold(),
            "away": df.loc[valid_mask, away_col].astype("string").str.strip().str.casefold(),
        })
        if league_col:
            key["league"] = df.loc[valid_mask, league_col].astype("string").str.strip().str.casefold()
        duplicate_rows = int(key.duplicated(keep="first").sum())
        print(f"Tekrar eden tarih + lig + ev + deplasman (tarihi dolu satırlarda): {duplicate_rows}")
        duplicate_report["duplicate_fixture_key_rows_valid_dates_only"] = duplicate_rows
        duplicate_report["rows_excluded_due_to_missing_date"] = int((~valid_mask).sum())
        report["duplicate_match_rows"] = duplicate_rows
    else:
        print("Tarih / ev sahibi / deplasman alanları tam bulunamadı; fikstür tekrar kontrolü atlandı.")
        report["duplicate_match_rows"] = None
    exact_dups = int(df.duplicated().sum())
    print(f"Birebir aynı tüm-sütun satır tekrarı: {exact_dups}")
    duplicate_report["exact_duplicate_rows"] = exact_dups
    report["duplicate_checks"] = duplicate_report

    print("\n=== ÖRNEK İLK 5 SATIR ===")
    print(df.head(5).to_string(index=False))
    report["sample_rows"] = df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records")

    expected = ["Premier League", "La Liga", "Bundesliga", "Serie A",
                "Ligue 1", "Süper Lig", "Primeira Liga", "Eredivisie"]
    if league_col:
        observed = set(df[league_col].dropna().astype(str).str.strip())
        report["expected_leagues_present_exact"] = {
            league: league in observed for league in expected
        }

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nJSON raporu yazıldı: {out}")


if __name__ == "__main__":
    main()
