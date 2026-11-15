"""Build an immutable, leakage-conscious normalized dataset artifact from both sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from zk2.data.adapters.data_csv import DataCsvAdapter
from zk2.data.adapters.epl_xlsm import EplXlsmAdapter

BASE_COLUMNS = [
    "_source_dataset", "_source_row", "_date", "_season", "_league_canonical",
    "_team_home_canonical", "_team_away_canonical", "home_score", "away_score",
    "_y_1x2", "_y_ou25", "_y_kg", "_kickoff_time",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}


def prepare_source(frame: pd.DataFrame, source_name: str, source_row_column: str) -> pd.DataFrame:
    out = frame.copy()
    out["_source_dataset"] = source_name
    if source_row_column not in out.columns:
        raise ValueError(f"Missing source row column for {source_name}: {source_row_column}")
    out["_source_row"] = out[source_row_column]
    if "_kickoff_time" not in out.columns:
        out["_kickoff_time"] = pd.NA
    optional = [
        column for column in out.columns
        if column.startswith("_odd_") or column.startswith("_market_") or column == "_asian_handicap_line"
    ]
    columns = BASE_COLUMNS + optional
    return out.reindex(columns=list(dict.fromkeys(columns)))


def build_locked_dataset(csv_path: Path, xlsm_path: Path, output_path: Path, manifest_path: Path) -> dict:
    csv_sha = sha256_file(csv_path)
    xlsm_sha = sha256_file(xlsm_path)
    tr_adapter = DataCsvAdapter(csv_path).load()
    epl_adapter = EplXlsmAdapter(xlsm_path).load()
    tr = tr_adapter.transform()
    epl = epl_adapter.transform()

    tr_audit = DataCsvAdapter.audit(tr)
    epl_audit = EplXlsmAdapter.audit(epl)
    tr_duplicate_rows = int(tr.duplicated(["_date_only", "_team_home_canonical", "_team_away_canonical"], keep=False).sum())
    epl_duplicate_rows = int(epl.duplicated(["_date_only", "_team_home_canonical", "_team_away_canonical"], keep=False).sum())
    for name, report, duplicate_rows in (("DATA.csv", tr_audit, tr_duplicate_rows), ("EPL.xlsm", epl_audit, epl_duplicate_rows)):
        if report["invalid_date_rows"] or report.get("missing_season_rows", 0) or report["missing_score_rows"] or report["missing_home_team_rows"] or report["missing_away_team_rows"]:
            raise ValueError(f"{name} failed core row-quality validation: {report}")
        if duplicate_rows:
            raise ValueError(f"{name} contains duplicate date/home/away match keys: {duplicate_rows}")

    tr_teams = set(tr["_team_home_canonical"]) | set(tr["_team_away_canonical"])
    epl_teams = set(epl["_team_home_canonical"]) | set(epl["_team_away_canonical"])
    team_overlap = tr_teams & epl_teams
    if team_overlap:
        raise ValueError(f"Cross-source canonical team overlap requires review: {sorted(team_overlap)}")

    tr_keys = set(zip(tr["_date_only"], tr["_team_home_canonical"], tr["_team_away_canonical"]))
    epl_keys = set(zip(epl["_date_only"], epl["_team_home_canonical"], epl["_team_away_canonical"]))
    match_overlap = tr_keys & epl_keys
    if match_overlap:
        raise ValueError(f"Cross-source exact match overlap requires review: {list(match_overlap)[:10]}")

    tr["_source_row"] = tr["_source_order"] + 2
    tr_prepared = prepare_source(tr, "DATA.csv", "_source_row")
    epl_prepared = prepare_source(epl, "EPL.xlsm", "_source_row")
    combined = pd.concat([tr_prepared, epl_prepared], ignore_index=True, sort=False)
    combined = combined.sort_values(["_date", "_source_dataset", "_source_row"], kind="stable").reset_index(drop=True)

    if combined[["_y_1x2", "_y_ou25", "_y_kg"]].isna().any().any():
        raise ValueError("At least one target is missing; locked dataset cannot include incomplete labels")
    duplicate_combined = combined.duplicated(["_date", "_team_home_canonical", "_team_away_canonical"], keep=False)
    if duplicate_combined.any():
        raise ValueError("Combined dataset contains duplicate match keys")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8", compression="gzip")
    odds_columns = [c for c in combined.columns if c.startswith("_odd_") or c.startswith("_market_") or c == "_asian_handicap_line"]
    manifest = {
        "dataset_id": "zk2-locked-v1",
        "created_from_zkbot_commit": os.environ.get("GITHUB_SHA"),
        "sources": {
            "DATA.csv": {
                "repository": "xsinanxaykacx-debug/trsuper",
                "path": "data/csv/DATA.csv",
                "sha256": csv_sha,
                "rows": int(len(tr)),
                "raw_columns": int(len(tr_adapter.df.columns)),
                "normalized_columns": int(len(tr.columns)),
                "date_min": str(tr["_date"].min()),
                "date_max": str(tr["_date"].max()),
                "canonical_teams": len(tr_teams),
                "target_counts": {target: target_counts(tr, target) for target in ("_y_1x2", "_y_ou25", "_y_kg")},
            },
            "EPL.xlsm": {
                "repository": "xsinanxaykacx-debug/zkbot",
                "path": "EPL.xlsm",
                "sha256": xlsm_sha,
                "rows": int(len(epl)),
                "raw_columns": int(len(epl_adapter.df.columns)),
                "normalized_columns": int(len(epl.columns)),
                "date_min": str(epl["_date"].min()),
                "date_max": str(epl["_date"].max()),
                "canonical_teams": len(epl_teams),
                "target_counts": {target: target_counts(epl, target) for target in ("_y_1x2", "_y_ou25", "_y_kg")},
                "coalesced_market_summary_columns": {
                    column: int(epl[column].notna().sum())
                    for column in epl.columns if column.startswith("_market_")
                },
            },
        },
        "validation": {
            "cross_source_team_overlap": 0,
            "cross_source_exact_match_overlap": 0,
            "combined_duplicate_match_rows": 0,
            "missing_targets": 0,
            "all_dates_valid": True,
            "all_scores_present": True,
            "all_team_identities_present": True,
        },
        "combined": {
            "rows": int(len(combined)),
            "columns": int(len(combined.columns)),
            "date_min": str(combined["_date"].min()),
            "date_max": str(combined["_date"].max()),
            "source_row_counts": {str(k): int(v) for k, v in combined["_source_dataset"].value_counts().items()},
            "canonical_leagues": sorted(str(x) for x in combined["_league_canonical"].unique()),
            "canonical_teams": len(set(combined["_team_home_canonical"]) | set(combined["_team_away_canonical"])),
            "feature_market_columns": odds_columns,
            "outcome_columns": ["home_score", "away_score", "_y_1x2", "_y_ou25", "_y_kg"],
            "file": output_path.name,
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
        },
        "policy": {
            "raw_source_files_not_committed_to_zkbot": True,
            "post_match_statistics_excluded_from_combined_table": True,
            "closing_odds_kept_separate_from_opening_odds": True,
            "missing_odds_remain_null": True,
            "target_labels_derived_from_final_scores": True,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_csv", type=Path)
    parser.add_argument("epl_xlsm", type=Path)
    parser.add_argument("--output", type=Path, default=Path("zk2_locked_dataset.csv.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("zk2_dataset_manifest.json"))
    args = parser.parse_args()
    manifest = build_locked_dataset(args.data_csv, args.epl_xlsm, args.output, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
