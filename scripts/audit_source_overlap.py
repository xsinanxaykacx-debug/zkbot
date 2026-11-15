"""Compare canonical team and exact match identities across DATA.csv and EPL.xlsm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from zk2.data.adapters.data_csv import DataCsvAdapter
from zk2.data.adapters.epl_xlsm import EplXlsmAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_csv", type=Path)
    parser.add_argument("epl_xlsm", type=Path)
    parser.add_argument("--output", type=Path, default=Path("source_overlap_report.json"))
    args = parser.parse_args()

    tr = DataCsvAdapter(args.data_csv).load().transform()
    epl = EplXlsmAdapter(args.epl_xlsm).load().transform()
    tr_teams = set(tr["_team_home_canonical"].dropna()) | set(tr["_team_away_canonical"].dropna())
    epl_teams = set(epl["_team_home_canonical"].dropna()) | set(epl["_team_away_canonical"].dropna())
    tr_keys = set(zip(tr["_date_only"], tr["_team_home_canonical"], tr["_team_away_canonical"]))
    epl_keys = set(zip(epl["_date_only"], epl["_team_home_canonical"], epl["_team_away_canonical"]))
    common_teams = sorted(tr_teams & epl_teams)
    common_matches = sorted(tr_keys & epl_keys, key=str)

    tr_audit = DataCsvAdapter.audit(tr)
    epl_audit = EplXlsmAdapter.audit(epl)
    report = {
        "data_csv_audit": tr_audit,
        "epl_xlsm_audit": {
            **epl_audit,
            "target_1x2_distribution": {str(k): int(v) for k, v in epl["_y_1x2"].value_counts(dropna=False).items()},
            "target_ou25_distribution": {str(k): int(v) for k, v in epl["_y_ou25"].value_counts(dropna=False).items()},
            "target_kg_distribution": {str(k): int(v) for k, v in epl["_y_kg"].value_counts(dropna=False).items()},
            "duplicate_header_columns_preserved": sum("__col_" in column for column in epl.columns),
            "missing_bet365_ou25_rows": int(epl[["_odd_over25_bet365_open", "_odd_under25_bet365_open"]].isna().any(axis=1).sum()),
            "missing_pinnacle_ou25_rows": int(epl[["_odd_over25_pinnacle_open", "_odd_under25_pinnacle_open"]].isna().any(axis=1).sum()),
        },
        "data_csv": {
            "rows": int(len(tr)),
            "date_min": str(tr["_date"].min()),
            "date_max": str(tr["_date"].max()),
            "unique_canonical_teams": len(tr_teams),
            "canonical_leagues": sorted(str(x) for x in tr["_league_canonical"].dropna().unique()),
            "invalid_dates": int((~tr["_is_valid_date"]).sum()),
        },
        "epl_xlsm": {
            "rows": int(len(epl)),
            "date_min": str(epl["_date"].min()),
            "date_max": str(epl["_date"].max()),
            "unique_canonical_teams": len(epl_teams),
            "canonical_leagues": sorted(str(x) for x in epl["_league_canonical"].dropna().unique()),
            "invalid_dates": int((~epl["_is_valid_date"]).sum()),
        },
        "team_overlap_count": len(common_teams),
        "common_canonical_teams": common_teams,
        "exact_match_key_overlap_count": len(common_matches),
        "exact_match_key_overlap_examples": [
            {"date": str(d), "home": home, "away": away} for d, home, away in common_matches[:20]
        ],
        "interpretation": "A zero team overlap is expected for Turkish domestic leagues versus English Premier League; do not merge teams across different identities based on spelling similarity.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
