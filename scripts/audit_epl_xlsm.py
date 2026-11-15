"""Read-only structural audit of EPL.xlsm. Macros are never executed."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from openpyxl import load_workbook


def clean(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def audit_workbook(path: Path) -> dict:
    # keep_vba preserves VBA parts in memory if the workbook is later saved;
    # data_only reads cached values. No macro code is executed.
    workbook = load_workbook(path, read_only=True, data_only=True, keep_vba=True)
    result = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sheet_names": workbook.sheetnames,
        "sheets": [],
        "macros_present": any(name.endswith("vbaProject.bin") for name in getattr(workbook, "_archive", []).namelist()) if hasattr(getattr(workbook, "_archive", None), "namelist") else None,
    }
    for sheet in workbook.worksheets:
        preview = []
        nonempty_by_column = {}
        date_values = []
        for row_number, row in enumerate(sheet.iter_rows(), start=1):
            values = [clean(cell.value) for cell in row]
            if any(value is not None and value != "" for value in values):
                if len(preview) < 20:
                    preview.append({"row": row_number, "values": values[:40]})
                for col_number, value in enumerate(values, start=1):
                    if value is not None and value != "":
                        key = str(col_number)
                        nonempty_by_column[key] = nonempty_by_column.get(key, 0) + 1
                    if isinstance(row[col_number - 1].value, (datetime, date)):
                        date_values.append(row[col_number - 1].value)
        result["sheets"].append({
            "title": sheet.title,
            "max_row": sheet.max_row,
            "max_column": sheet.max_column,
            "nonempty_rows_preview": preview,
            "nonempty_cells_by_column_index": nonempty_by_column,
            "date_min": min(date_values).isoformat() if date_values else None,
            "date_max": max(date_values).isoformat() if date_values else None,
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
