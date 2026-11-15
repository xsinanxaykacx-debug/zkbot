"""Audit every raw team label in DATA.csv against the reviewed alias table."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from zk2.data.team_mapping import TEAM_CANONICAL, normalize_team, team_key


def audit_teams(path: Path) -> dict:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = ["Ev_Sahibi", "Deplasman"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing team columns: {missing}")

    counts: Counter[str] = Counter()
    for column in required:
        values = frame[column].dropna().astype(str).str.strip()
        counts.update(value for value in values if value)

    records = []
    unmapped = []
    for raw, count in sorted(counts.items(), key=lambda item: (-item[1], team_key(item[0]))):
        key = team_key(raw)
        canonical = normalize_team(raw)
        is_mapped = key in TEAM_CANONICAL
        record = {
            "raw_name": raw,
            "comparison_key": key,
            "canonical_name": canonical,
            "mapped_by_alias": is_mapped,
            "appearances_home_and_away": int(count),
        }
        records.append(record)
        if not is_mapped:
            unmapped.append(record)

    home_unique = int(frame["Ev_Sahibi"].dropna().astype(str).str.strip().replace("", pd.NA).nunique())
    away_unique = int(frame["Deplasman"].dropna().astype(str).str.strip().replace("", pd.NA).nunique())
    canonical_unique = {normalize_team(name) for name in counts}
    return {
        "source": str(path),
        "rows": int(len(frame)),
        "unique_raw_home_names": home_unique,
        "unique_raw_away_names": away_unique,
        "unique_raw_names_union": len(counts),
        "unique_canonical_names_after_current_aliases": len(canonical_unique),
        "raw_names_mapped_by_alias": sum(1 for name in counts if team_key(name) in TEAM_CANONICAL),
        "raw_names_without_explicit_alias_rule": len(unmapped),
        "names_without_explicit_alias_rule": unmapped,
        "all_team_name_records": records,
        "note": "Names without an explicit alias rule remain their own canonical identity. Review possible historical renames separately; never merge by fuzzy similarity.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("team_audit_report.json"))
    args = parser.parse_args()
    report = audit_teams(args.csv_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("all_team_name_records", "names_without_explicit_alias_rule")}, ensure_ascii=False, indent=2))
    print("\nTEAM LABELS WITHOUT AN EXPLICIT ALIAS RULE:")
    for item in report["names_without_explicit_alias_rule"]:
        print(f'{item["raw_name"]}\t{item["appearances_home_and_away"]}')


if __name__ == "__main__":
    main()
