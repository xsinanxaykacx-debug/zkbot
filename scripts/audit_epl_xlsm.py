"""Read-only structural and data-quality audit of EPL.xlsm. Macros are never executed."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from openpyxl import load_workbook

KEY_ODDS = [
    "Bet365_Ev", "Bet365_Beraberlik", "Bet365_Dep",
    "Pinnacle_Ev", "Pinnacle_Beraberlik", "Pinnacle_Dep",
    "EnYüksekEvOranı", "EnYüksekBeraberlikOranı", "EnYüksekDepOranı",
    "Bet365_Üst2,5", "Bet365_Alt2,5", "Pinnacle_Üst2,5", "Pinnacle_Alt2,5",
    "EnYüksekÜst2,5", "EnYüksekAlt2,5", "PinnacleKapanış_Ev",
    "PinnacleKapanış_Beraberlik", "PinnacleKapanış_Dep",
]


def clean(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def parse_date(value):
    if isinstance(value, (datetime, date)):
        return value.date() if isinstance(value, datetime) else value
    if isinstance(value, str):
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                pass
    return None


def audit_workbook(path: Path) -> dict:
    workbook = load_workbook(path, read_only=True, data_only=True, keep_vba=True)
    archive = getattr(workbook, "_archive", None)
    archive_names = archive.namelist() if archive is not None and hasattr(archive, "namelist") else []
    result = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sheet_names": workbook.sheetnames,
        "macros_present": any(name.endswith("vbaProject.bin") for name in archive_names),
        "sheets": [],
    }
    for sheet in workbook.worksheets:
        iterator = sheet.iter_rows(values_only=True)
        header = next(iterator, ())
        headers = [str(value).strip() if value is not None else None for value in header]
        counts = Counter()
        column_counts = Counter()
        team_counts = Counter()
        invalid_date_samples = []
        dates = []
        first_rows, last_rows = [], []
        nonempty_rows = 0
        missing = Counter()
        duplicate_keys = Counter()
        score_values = []
        for row_number, row in enumerate(iterator, start=2):
            if not any(value is not None and value != "" for value in row):
                continue
            nonempty_rows += 1
            record = {headers[i]: row[i] for i in range(min(len(headers), len(row))) if headers[i]}
            if len(first_rows) < 5:
                first_rows.append({"row": row_number, "values": [clean(v) for v in row]})
            last_rows.append({"row": row_number, "values": [clean(v) for v in row]})
            if len(last_rows) > 5:
                last_rows.pop(0)
            for index, value in enumerate(row):
                if value is not None and value != "" and index < len(headers) and headers[index]:
                    counts[headers[index]] += 1
                    column_counts[index + 1] += 1
            if "Tarih" in record:
                parsed = parse_date(record["Tarih"])
                if parsed:
                    dates.append(parsed)
                else:
                    missing["invalid_date"] += 1
                    if len(invalid_date_samples) < 25:
                        invalid_date_samples.append(clean(record["Tarih"]))
            for col in ("EvSahibi", "Deplasman"):
                value = record.get(col)
                if value is None or str(value).strip() == "":
                    missing[f"missing_{col}"] += 1
                else:
                    team_counts[str(value).strip()] += 1
            for col in ("MS_EvGol", "MS_DepGol"):
                if record.get(col) is None:
                    missing[f"missing_{col}"] += 1
            if record.get("MS_EvGol") is not None and record.get("MS_DepGol") is not None:
                score_values.append((record["MS_EvGol"], record["MS_DepGol"]))
            key = (clean(parse_date(record.get("Tarih"))), record.get("EvSahibi"), record.get("Deplasman"))
            duplicate_keys[key] += 1
        duplicates = sum(count for count in duplicate_keys.values() if count > 1)
        key_columns = {name: counts.get(name, 0) for name in KEY_ODDS if name in headers}
        result["sheets"].append({
            "title": sheet.title,
            "max_row": sheet.max_row,
            "max_column": sheet.max_column,
            "data_rows_nonempty": nonempty_rows,
            "headers": headers,
            "duplicate_headers": {name: [i + 1 for i, value in enumerate(headers) if value == name] for name, count in Counter(h for h in headers if h).items() if count > 1},
            "nonempty_cells_by_header": dict(counts),
            "nonempty_cells_by_column_index": {str(index): int(count) for index, count in column_counts.items()},
            "invalid_date_examples": invalid_date_samples,
            "date_min": min(dates).isoformat() if dates else None,
            "date_max": max(dates).isoformat() if dates else None,
            "invalid_or_missing_date_rows": int(missing["invalid_date"]),
            "missing_home_team_rows": int(missing["missing_EvSahibi"]),
            "missing_away_team_rows": int(missing["missing_Deplasman"]),
            "missing_home_score_rows": int(missing["missing_MS_EvGol"]),
            "missing_away_score_rows": int(missing["missing_MS_DepGol"]),
            "duplicate_date_home_away_rows": int(duplicates),
            "unique_team_names": len(team_counts),
            "team_name_counts": dict(team_counts),
            "key_odds_nonempty_counts": key_columns,
            "first_data_rows": first_rows,
            "last_data_rows": last_rows,
        })
    workbook.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("epl_xlsm_audit.json"))
    args = parser.parse_args()
    report = audit_workbook(args.workbook)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
