"""Full row-level audit for the public DATA.csv source; no rows are removed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from zk2.data.adapters.data_csv import DataCsvAdapter, OU25_NEW, OU25_OLD


def _numeric(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(values.replace({"": pd.NA, "nan": pd.NA}), errors="coerce")


def build_report(adapter: DataCsvAdapter, transformed: pd.DataFrame) -> dict:
    raw = adapter.df
    if raw is None:
        raise RuntimeError("Adapter must be loaded before building the report")

    def filled(column: str) -> int:
        return int(raw[column].notna().sum()) if column in raw else 0

    def valid_numeric_count(column: str) -> int:
        if column not in raw:
            return 0
        return int(_numeric(raw[column]).notna().sum())

    old_under, old_over = OU25_OLD
    new_under, new_over = OU25_NEW
    old_mask = _numeric(raw[old_under]).notna() if old_under in raw else pd.Series(False, index=raw.index)
    new_mask = _numeric(raw[new_under]).notna() if new_under in raw else pd.Series(False, index=raw.index)
    old_over_mask = _numeric(raw[old_over]).notna() if old_over in raw else pd.Series(False, index=raw.index)
    new_over_mask = _numeric(raw[new_over]).notna() if new_over in raw else pd.Series(False, index=raw.index)

    report = {
        "source_file": str(adapter.path),
        "rows": int(len(raw)),
        "columns": int(len(raw.columns)),
        "transformed_rows": int(len(transformed)),
        "rows_dropped_by_adapter": int(len(raw) - len(transformed)),
        "date": {
            "valid": int(transformed["_is_valid_date"].sum()),
            "invalid": int((~transformed["_is_valid_date"]).sum()),
            "min": transformed["_date"].min(),
            "max": transformed["_date"].max(),
        },
        "season": {
            "valid": int(transformed["_is_valid_season"].sum()),
            "invalid": int((~transformed["_is_valid_season"]).sum()),
            "counts": {str(k): int(v) for k, v in transformed["_season"].value_counts(dropna=False).items()},
        },
        "league_counts_raw": {str(k): int(v) for k, v in transformed["_league"].value_counts(dropna=False).items()},
        "league_counts_canonical": {str(k): int(v) for k, v in transformed["_league_canonical"].value_counts(dropna=False).items()},
        "scores": {
            "missing_home": int(transformed["MSE"].isna().sum()),
            "missing_away": int(transformed["MSD"].isna().sum()),
            "missing_either": int((transformed["MSE"].isna() | transformed["MSD"].isna()).sum()),
        },
        "teams": {
            "missing_home": int(transformed["_team_home_canonical"].isna().sum()),
            "missing_away": int(transformed["_team_away_canonical"].isna().sum()),
            "missing_either": int((transformed["_team_home_canonical"].isna() | transformed["_team_away_canonical"].isna()).sum()),
        },
        "odds": {
            "1x2_missing_any": int(transformed[["_odd_H", "_odd_D", "_odd_A"]].isna().any(axis=1).sum()),
            "1x2_missing_all": int(transformed[["_odd_H", "_odd_D", "_odd_A"]].isna().all(axis=1).sum()),
            "ou25_missing_any": int(transformed[["_odd_under25", "_odd_over25"]].isna().any(axis=1).sum()),
            "ou25_missing_all": int(transformed[["_odd_under25", "_odd_over25"]].isna().all(axis=1).sum()),
            "kg_missing_any": int(transformed[["_odd_kg_var", "_odd_kg_yok"]].isna().any(axis=1).sum()),
            "kg_missing_all": int(transformed[["_odd_kg_var", "_odd_kg_yok"]].isna().all(axis=1).sum()),
            "nonpositive_values": {
                col: int((_numeric(transformed[col]).le(0)).sum())
                for col in ("_odd_H", "_odd_D", "_odd_A", "_odd_under25", "_odd_over25", "_odd_kg_var", "_odd_kg_yok")
            },
        },
        "ou25_source_columns": {
            "old_under_filled": int(old_mask.sum()),
            "old_over_filled": int(old_over_mask.sum()),
            "new_under_filled": int(new_mask.sum()),
            "new_over_filled": int(new_over_mask.sum()),
            "under_both_filled": int((old_mask & new_mask).sum()),
            "over_both_filled": int((old_over_mask & new_over_mask).sum()),
            "under_old_only": int((old_mask & ~new_mask).sum()),
            "under_new_only": int((new_mask & ~old_mask).sum()),
        },
        "targets": {
            "1x2": {str(k): int(v) for k, v in transformed["_y_1x2"].value_counts(dropna=False).items()},
            "ou25": {str(k): int(v) for k, v in transformed["_y_ou25"].value_counts(dropna=False).items()},
            "kg": {str(k): int(v) for k, v in transformed["_y_kg"].value_counts(dropna=False).items()},
        },
        "identifiers": {
            "missing_match_id": int(transformed["_match_id"].isna().sum()),
            "duplicate_match_id_rows": int(transformed["_match_id"].duplicated(keep=False).sum()),
            "duplicate_date_home_away_rows": int(transformed.duplicated(["_date_only", "_team_home_canonical", "_team_away_canonical"], keep=False).sum()),
        },
        "raw_numeric_parse_counts": {
            col: valid_numeric_count(col)
            for col in ("MSE", "MSD", "Maç_Sonucu_|_1", "Maç_Sonucu_|_X", "Maç_Sonucu_|_2", *OU25_OLD, *OU25_NEW, "Karşılıklı_Gol_|_Var", "Karşılıklı_Gol_|_Yok")
        },
        "raw_nonempty_counts": {
            col: filled(col)
            for col in ("Tarih", "Lig/Sezon", "Maç_ID", "Ev_Sahibi", "Deplasman", "MSE", "MSD", *OU25_OLD, *OU25_NEW)
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data_audit_report.json"))
    args = parser.parse_args()

    adapter = DataCsvAdapter(args.csv_path).load()
    transformed = adapter.transform()
    report = build_report(adapter, transformed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
