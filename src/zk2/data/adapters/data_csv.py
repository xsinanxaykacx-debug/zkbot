"""Adapter for the Turkish DATA.csv source.

The adapter preserves every source row. Invalid dates/seasons are exposed for
validation rather than silently dropped. Raw post-match columns remain in the
frame, so callers must select only pre-match features for model training.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from zk2.data.targets import add_targets
from zk2.data.team_mapping import normalize_team


REQUIRED_COLUMNS = (
    "Tarih",
    "Lig/Sezon",
    "Maç_ID",
    "Ev_Sahibi",
    "Deplasman",
    "MSE",
    "MSD",
    "Maç_Sonucu_|_1",
    "Maç_Sonucu_|_X",
    "Maç_Sonucu_|_2",
    "Karşılıklı_Gol_|_Var",
    "Karşılıklı_Gol_|_Yok",
)

OU25_OLD = ("AÜ_2,5_Gol_|_Alt", "AÜ_2,5_Gol_|_Üst")
OU25_NEW = ("2,5_Alt/Üst_|_Alt", "2,5_Alt/Üst_|_Üst")


def _numeric_series(series: pd.Series) -> pd.Series:
    """Parse numeric strings, including Turkish decimal-comma notation."""
    values = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    values = values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(values, errors="coerce")


def load_ou25_odds(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Combine legacy and current OU2.5 columns; legacy values take priority."""
    pairs = (OU25_OLD, OU25_NEW)
    for under_col, over_col in pairs:
        present = (under_col in frame.columns, over_col in frame.columns)
        if present[0] != present[1]:
            raise ValueError(f"OU2.5 column pair is incomplete: {under_col}, {over_col}")

    under = pd.Series(float("nan"), index=frame.index, dtype="float64")
    over = pd.Series(float("nan"), index=frame.index, dtype="float64")
    if all(column in frame.columns for column in OU25_OLD):
        under = _numeric_series(frame[OU25_OLD[0]])
        over = _numeric_series(frame[OU25_OLD[1]])
    if all(column in frame.columns for column in OU25_NEW):
        under = under.fillna(_numeric_series(frame[OU25_NEW[0]]))
        over = over.fillna(_numeric_series(frame[OU25_NEW[1]]))
    if not any(all(column in frame.columns for column in pair) for pair in pairs):
        raise ValueError("No complete OU2.5 odds column pair found")
    return under, over


class DataCsvAdapter:
    """Load and normalize the DATA.csv source without discarding source rows."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.df: pd.DataFrame | None = None

    def load(self) -> "DataCsvAdapter":
        self.df = pd.read_csv(self.path, encoding="utf-8-sig", low_memory=False)
        return self

    def transform(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        source = self.df if frame is None else frame
        if source is None:
            raise RuntimeError("Call load() first or pass a source DataFrame to transform()")

        missing = [column for column in REQUIRED_COLUMNS if column not in source.columns]
        if missing:
            raise ValueError(f"DATA.csv is missing required columns: {missing}")

        out = source.copy()
        raw_date = out["Tarih"].astype("string").str.strip()
        out["_date"] = pd.to_datetime(raw_date, format="%d-%m-%Y", errors="coerce")
        out["_date_only"] = out["_date"].dt.normalize()
        out["_is_valid_date"] = out["_date"].notna()

        league_season = out["Lig/Sezon"].astype("string").str.strip()
        season = league_season.str.extract(r"(\d{4}/\d{4})", expand=False)
        out["_season"] = season
        out["_is_valid_season"] = out["_season"].notna()
        out["_league"] = league_season.str.replace(
            r"\s*\d{4}/\d{4}\s*$", "", regex=True
        ).str.strip()
        out["_league_canonical"] = out["_league"]

        out["_match_id"] = out["Maç_ID"].astype("string").str.strip()
        out["_team_home_canonical"] = out["Ev_Sahibi"].map(
            lambda value: pd.NA if pd.isna(value) or not str(value).strip() else normalize_team(value)
        )
        out["_team_away_canonical"] = out["Deplasman"].map(
            lambda value: pd.NA if pd.isna(value) or not str(value).strip() else normalize_team(value)
        )

        out["MSE"] = _numeric_series(out["MSE"])
        out["MSD"] = _numeric_series(out["MSD"])
        out["home_score"] = out["MSE"]
        out["away_score"] = out["MSD"]
        out = add_targets(out, "home_score", "away_score")

        out["_odd_under25"], out["_odd_over25"] = load_ou25_odds(out)
        for source_col, target_col in (
            ("Maç_Sonucu_|_1", "_odd_H"),
            ("Maç_Sonucu_|_X", "_odd_D"),
            ("Maç_Sonucu_|_2", "_odd_A"),
            ("Karşılıklı_Gol_|_Var", "_odd_kg_var"),
            ("Karşılıklı_Gol_|_Yok", "_odd_kg_yok"),
        ):
            out[target_col] = _numeric_series(out[source_col])

        if "Saat" in out.columns:
            out["_kickoff_time"] = out["Saat"].astype("string").str.strip()
        else:
            out["_kickoff_time"] = pd.NA

        # Stable sort; invalid dates remain at the end for the audit report.
        out["_source_order"] = range(len(out))
        out = out.sort_values(
            ["_date_only", "_kickoff_time", "_match_id", "_source_order"],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)
        return out

    @staticmethod
    def audit(frame: pd.DataFrame) -> dict[str, int]:
        """Return basic row-quality counts without mutating or filtering data."""
        required_outputs = ("_date", "_season", "_team_home_canonical", "_team_away_canonical")
        missing_outputs = [column for column in required_outputs if column not in frame.columns]
        if missing_outputs:
            raise ValueError(f"Transformed frame is missing audit columns: {missing_outputs}")
        return {
            "rows": int(len(frame)),
            "invalid_date_rows": int(frame["_date"].isna().sum()),
            "missing_season_rows": int(frame["_season"].isna().sum()),
            "missing_home_team_rows": int((frame["_team_home_canonical"].isna() | frame["_team_home_canonical"].astype("string").str.strip().eq("")).sum()),
            "missing_away_team_rows": int((frame["_team_away_canonical"].isna() | frame["_team_away_canonical"].astype("string").str.strip().eq("")).sum()),
            "missing_score_rows": int((frame["MSE"].isna() | frame["MSD"].isna()).sum()),
            "missing_1x2_odds_rows": int(frame[["_odd_H", "_odd_D", "_odd_A"]].isna().any(axis=1).sum()),
            "missing_ou25_odds_rows": int(frame[["_odd_under25", "_odd_over25"]].isna().any(axis=1).sum()),
            "missing_kg_odds_rows": int(frame[["_odd_kg_var", "_odd_kg_yok"]].isna().any(axis=1).sum()),
        }
