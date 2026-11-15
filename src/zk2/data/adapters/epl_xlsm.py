"""Adapter for the EPL.xlsm DATA worksheet; never executes macros or reads RAPOR."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from zk2.data.epl_team_mapping import normalize_epl_team
from zk2.data.targets import add_targets

SHEET_NAME = "DATA"
REQUIRED_COLUMNS = ("Tarih", "EvSahibi", "Deplasman", "MS_EvGol", "MS_DepGol")
DUPLICATE_SUMMARY_COLUMNS = {
    "_market_best_home_1x2": ("EnYüksekEvOranı__col_46", "EnYüksekEvOranı__col_67"),
    "_market_average_home_1x2": ("OrtalamaEvOranı__col_47", "OrtalamaEvOranı__col_70"),
    "_market_best_draw_1x2": ("EnYüksekBeraberlikOranı__col_48", "EnYüksekBeraberlikOranı__col_68"),
    "_market_average_draw_1x2": ("OrtalamaBeraberlikOranı__col_49", "OrtalamaBeraberlikOranı__col_71"),
    "_market_best_away_1x2": ("EnYüksekDepOranı__col_50", "EnYüksekDepOranı__col_69"),
    "_market_average_away_1x2": ("OrtalamaDepOranı__col_51", "OrtalamaDepOranı__col_72"),
    "_market_best_over25": ("EnYüksekÜst2,5__col_53", "EnYüksekÜst2,5__col_77"),
    "_market_average_over25": ("OrtalamaÜst2,5__col_54", "OrtalamaÜst2,5__col_79"),
    "_market_best_under25": ("EnYüksekAlt2,5__col_55", "EnYüksekAlt2,5__col_78"),
    "_market_average_under25": ("OrtalamaAlt2,5__col_56", "OrtalamaAlt2,5__col_80"),
    "_asian_handicap_line": ("AsyaHandikapı__col_58", "AsyaHandikapı__col_81"),
    "_market_best_home_ah": ("EnYüksekEvHandikapOranı__col_59", "EnYüksekEvHandikapOranı__col_86"),
    "_market_average_home_ah": ("OrtalamaEvHandikapOranı__col_60", "OrtalamaEvHandikapOranı__col_88"),
    "_market_best_away_ah": ("EnYüksekDepHandikapOranı__col_61", "EnYüksekDepHandikapOranı__col_87"),
    "_market_average_away_ah": ("OrtalamaDepHandikapOranı__col_62", "OrtalamaDepHandikapOranı__col_89"),
}


ODDS_COLUMNS = {
    "Bet365_Ev": "_odd_H_bet365_open",
    "Bet365_Beraberlik": "_odd_D_bet365_open",
    "Bet365_Dep": "_odd_A_bet365_open",
    "Pinnacle_Ev": "_odd_H_pinnacle_open",
    "Pinnacle_Beraberlik": "_odd_D_pinnacle_open",
    "Pinnacle_Dep": "_odd_A_pinnacle_open",
    "Bet365_Üst2,5": "_odd_over25_bet365_open",
    "Bet365_Alt2,5": "_odd_under25_bet365_open",
    "Pinnacle_Üst2,5": "_odd_over25_pinnacle_open",
    "Pinnacle_Alt2,5": "_odd_under25_pinnacle_open",
    "Bet365_Kapanış_Ev": "_odd_H_bet365_close",
    "Bet365_Kapanış_Beraberlik": "_odd_D_bet365_close",
    "Bet365_Kapanış_Dep": "_odd_A_bet365_close",
    "PinnacleKapanış_Ev": "_odd_H_pinnacle_close",
    "PinnacleKapanış_Beraberlik": "_odd_D_pinnacle_close",
    "PinnacleKapanış_Dep": "_odd_A_pinnacle_close",
    "Bet365_Kapanış_Üst2,5": "_odd_over25_bet365_close",
    "Bet365_Kapanış_Alt2,5": "_odd_under25_bet365_close",
    "Pinnacle_Kapanış_Üst2,5": "_odd_over25_pinnacle_close",
    "Pinnacle_Kapanış_Alt2,5": "_odd_under25_pinnacle_close",
}


def _parse_date(value: object):
    if isinstance(value, datetime):
        return pd.Timestamp(value.date())
    if isinstance(value, date):
        return pd.Timestamp(value)
    if value is None:
        return pd.NaT
    raw = str(value).strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return pd.Timestamp(datetime.strptime(raw, fmt).date())
        except ValueError:
            continue
    return pd.NaT


def _numeric(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}), errors="coerce")


def coalesce_duplicate_numeric_columns(
    frame: pd.DataFrame, first_column: str, second_column: str
) -> pd.Series:
    """Merge disjoint/identical numeric source columns; reject conflicting overlaps."""
    first = _numeric(frame[first_column])
    second = _numeric(frame[second_column])
    both = first.notna() & second.notna()
    if both.any():
        equal = pd.Series(
            np.isclose(first.loc[both].to_numpy(), second.loc[both].to_numpy(), rtol=1e-9, atol=1e-12),
            index=first.loc[both].index,
        )
        if not bool(equal.all()):
            raise ValueError(
                f"Conflicting values in duplicate source columns: {first_column}, {second_column}"
            )
    return first.combine_first(second)


def _unique_headers(values: tuple[object, ...]) -> list[str]:
    names = [str(value).strip() if value is not None else "" for value in values]
    counts = {name: names.count(name) for name in set(names) if name}
    result = []
    for index, name in enumerate(names, start=1):
        if not name:
            result.append(f"_empty_header_col_{index}")
        elif counts.get(name, 0) > 1:
            result.append(f"{name}__col_{index}")
        else:
            result.append(name)
    return result


class EplXlsmAdapter:
    """Read DATA sheet, preserve duplicate headers by physical column index."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.df: pd.DataFrame | None = None

    def load(self) -> "EplXlsmAdapter":
        workbook = load_workbook(self.path, read_only=True, data_only=True, keep_vba=False)
        if SHEET_NAME not in workbook.sheetnames:
            workbook.close()
            raise ValueError(f"Workbook is missing required worksheet: {SHEET_NAME}")
        sheet = workbook[SHEET_NAME]
        iterator = sheet.iter_rows(values_only=True)
        header = next(iterator, None)
        if header is None:
            workbook.close()
            raise ValueError("EPL DATA worksheet is empty")
        columns = _unique_headers(tuple(header))
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            workbook.close()
            raise ValueError(f"EPL DATA worksheet is missing required columns: {missing}")
        rows = [row for row in iterator if any(value is not None and value != "" for value in row)]
        workbook.close()
        self.df = pd.DataFrame(rows, columns=columns)
        return self

    def transform(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        source = self.df if frame is None else frame
        if source is None:
            raise RuntimeError("Call load() first or pass a source DataFrame to transform()")
        missing = [name for name in REQUIRED_COLUMNS if name not in source.columns]
        if missing:
            raise ValueError(f"EPL DATA frame is missing required columns: {missing}")
        out = source.copy()
        out["_source_order"] = range(len(out))
        out["_source_sheet"] = SHEET_NAME
        out["_source_row"] = out["_source_order"] + 2
        out["_date"] = out["Tarih"].map(_parse_date)
        out["_date_only"] = out["_date"]
        out["_is_valid_date"] = out["_date"].notna()
        out["_season"] = out["_date"].map(
            lambda value: f"{value.year if value.month >= 8 else value.year - 1}/"
            f"{(value.year if value.month >= 8 else value.year - 1) + 1}"
            if pd.notna(value) else pd.NA
        )
        out["_is_valid_season"] = out["_season"].notna()
        out["_league"] = "England Premier League"
        out["_league_canonical"] = "England Premier League"
        out["_team_home_canonical"] = out["EvSahibi"].map(
            lambda value: pd.NA if pd.isna(value) or not str(value).strip() else normalize_epl_team(value)
        )
        out["_team_away_canonical"] = out["Deplasman"].map(
            lambda value: pd.NA if pd.isna(value) or not str(value).strip() else normalize_epl_team(value)
        )
        out["home_score"] = _numeric(out["MS_EvGol"])
        out["away_score"] = _numeric(out["MS_DepGol"])
        out = add_targets(out, "home_score", "away_score")
        for target_col, (first_col, second_col) in DUPLICATE_SUMMARY_COLUMNS.items():
            if first_col in out.columns and second_col in out.columns:
                out[target_col] = coalesce_duplicate_numeric_columns(out, first_col, second_col)
        for source_col, target_col in ODDS_COLUMNS.items():
            if source_col in out.columns:
                out[target_col] = _numeric(out[source_col])
        if "Saat" in out.columns:
            out["_kickoff_time"] = out["Saat"].astype("string").str.strip()
        else:
            out["_kickoff_time"] = pd.NA
        out = out.sort_values(
            ["_date_only", "_kickoff_time", "_source_order"],
            kind="stable", na_position="last"
        ).reset_index(drop=True)
        return out

    @staticmethod
    def audit(frame: pd.DataFrame) -> dict[str, int]:
        required = ("_date", "_season", "_team_home_canonical", "_team_away_canonical", "_y_1x2")
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Transformed EPL frame is missing audit columns: {missing}")
        return {
            "rows": int(len(frame)),
            "invalid_date_rows": int(frame["_date"].isna().sum()),
            "missing_season_rows": int(frame["_season"].isna().sum()),
            "missing_home_team_rows": int(frame["_team_home_canonical"].isna().sum()),
            "missing_away_team_rows": int(frame["_team_away_canonical"].isna().sum()),
            "missing_score_rows": int((frame["home_score"].isna() | frame["away_score"].isna()).sum()),
            "duplicate_date_home_away_rows": int(frame.duplicated(["_date_only", "_team_home_canonical", "_team_away_canonical"], keep=False).sum()),
            "missing_pinnacle_1x2_rows": int(frame[[c for c in ("_odd_H_pinnacle_open", "_odd_D_pinnacle_open", "_odd_A_pinnacle_open") if c in frame]].isna().any(axis=1).sum()),
        }
