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

    if date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        valid = parsed.dropna()
        print("\n=== TARİH KAPSAMI ===")
        print(f"İlk tarih: {valid.min() if not valid.empty else 'YOK'}")
        print(f"Son tarih: {valid.max() if not valid.empty else 'YOK'}")
        print(f"Geçersiz/boş tarih: {int(parsed.isna().sum())}")
        report["date_min"] = str(valid.min()) if not valid.empty else None
        report["date_max"] = str(valid.max()) if not valid.empty else None
        report["invalid_dates"] = int(parsed.isna().sum())
    else:
        print("\nUYARI: Tarih sütunu otomatik tanınamadı.")

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
    if date_col and home_col and away_col:
        key = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d"),
            "home": df[home_col].astype("string").str.strip().str.casefold(),
            "away": df[away_col].astype("string").str.strip().str.casefold(),
        })
        if league_col:
            key["league"] = df[league_col].astype("string").str.strip().str.casefold()
        duplicate_rows = int(key.duplicated(keep="first").sum())
        print(f"Tekrar eden maç satırı: {duplicate_rows}")
        report["duplicate_match_rows"] = duplicate_rows
    else:
        print("Tarih / ev sahibi / deplasman alanları tam bulunamadı; tekrar kontrolü atlandı.")
        report["duplicate_match_rows"] = None

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
