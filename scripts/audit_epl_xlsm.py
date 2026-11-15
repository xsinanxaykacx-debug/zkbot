"""Read-only structural audit of EPL.xlsm. Macros are never executed."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from openpyxl import load_workbook


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
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                pass
    return None


def audit_workbook(path: Path) -> dict:
    # openpyxl reads cached values; it never executes VBA or Excel formulas.
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
        date_values = []
        first_rows = []
        nonempty_rows = 0
        for row_number, row in enumerate(iterator, start=2):
            if any(value is not None and value != "" for value in row):
                nonempty_rows += 1
                if len(first_rows) < 10:
                    first_rows.append({"row": row_number, "values": [clean(v) for v in row]})
            for index, value in enumerate(row):
                if value is not None and value != "" and index < len(headers) and headers[index]:
                    counts[headers[index]] += 1
            if row:
                parsed = parse_date(row[0])
                if parsed:
                    date_values.append(parsed)
        result["sheets"].append({
            "title": sheet.title,
            "max_row": sheet.max_row,
            "max_column": sheet.max_column,
            "data_rows_nonempty": nonempty_rows,
            "headers": headers,
            "nonempty_cells_by_header": dict(counts),
            "date_min": min(date_values).isoformat() if date_values else None,
            "date_max": max(date_values).isoformat() if date_values else None,
            "first_data_rows": first_rows,
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
