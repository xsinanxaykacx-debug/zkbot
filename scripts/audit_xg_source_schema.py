"""Audit existing source files for match-level expected-goals fields.

This is a schema/coverage audit of user-owned source files only. It does not
contact FotMob or any other third-party service.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook

XG_HEADER = re.compile(
    r"(?:^|[^a-z])xg(?:$|[^a-z])|expected[ _-]*goals?",
    flags=re.IGNORECASE,
)


def is_xg_header(value: object) -> bool:
    if value is None:
        return False
    return bool(XG_HEADER.search(str(value).strip()))


def _coverage(values: list[object]) -> dict[str, Any]:
    nonempty = [value for value in values if not pd.isna(value) and str(value).strip() != ""]
    numeric = pd.to_numeric(pd.Series(nonempty, dtype="object"), errors="coerce")
    return {
        "non_null_rows": int(len(nonempty)),
        "numeric_rows": int(numeric.notna().sum()),
        "non_null_rate": float(len(nonempty) / len(values)) if values else 0.0,
        "numeric_rate": float(numeric.notna().sum() / len(values)) if values else 0.0,
    }


def audit_csv(path: Path, source_name: str = "DATA.csv") -> dict[str, Any]:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    candidates = []
    for column in frame.columns:
        if is_xg_header(column):
            coverage = _coverage(frame[column].tolist())
            candidates.append({
                "column": str(column),
                "column_index_0based": int(frame.columns.get_loc(column)),
                **coverage,
            })
    return {
        "source": source_name,
        "path": path.name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "xg_candidate_columns": candidates,
        "xg_candidate_column_count": len(candidates),
    }


def audit_xlsm(path: Path, sheet_name: str = "DATA") -> dict[str, Any]:
    workbook = load_workbook(path, read_only=True, data_only=True, keep_vba=False)
    try:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f"Workbook is missing required worksheet: {sheet_name}")
        sheet = workbook[sheet_name]
        iterator = sheet.iter_rows(values_only=True)
        headers = next(iterator, None)
        if headers is None:
            raise ValueError(f"Worksheet {sheet_name} is empty")
        candidate_indices = [
            index for index, header in enumerate(headers) if is_xg_header(header)
        ]
        values_by_column = {index: [] for index in candidate_indices}
        row_count = 0
        for row in iterator:
            if not any(value is not None and value != "" for value in row):
                continue
            row_count += 1
            for index in candidate_indices:
                values_by_column[index].append(row[index] if index < len(row) else None)
        candidates = []
        for index in candidate_indices:
            header = headers[index]
            candidates.append({
                "column": str(header),
                "column_index_0based": int(index),
                **_coverage(values_by_column[index]),
            })
        return {
            "source": "EPL.xlsm",
            "path": path.name,
            "sheet": sheet_name,
            "rows": int(row_count),
            "columns": int(len(headers)),
            "xg_candidate_columns": candidates,
            "xg_candidate_column_count": len(candidates),
        }
    finally:
        workbook.close()


def audit_sources(csv_path: Path, xlsm_path: Path) -> dict[str, Any]:
    sources = [
        audit_csv(csv_path),
        audit_xlsm(xlsm_path),
    ]
    return {
        "purpose": (
            "Schema and non-null coverage audit of existing ZK source files. "
            "This is not a FotMob match-coverage audit and makes no inference "
            "about third-party provider coverage."
        ),
        "sources": sources,
        "conclusion": (
            "No candidate xG fields were found in either source."
            if not any(source["xg_candidate_column_count"] for source in sources)
            else "Candidate xG fields were found; review names and values before mapping."
        ),
        "missing_value_policy": "Missing values remain missing; no zero imputation is performed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_csv", type=Path)
    parser.add_argument("epl_xlsm", type=Path)
    parser.add_argument("--json", type=Path, default=Path("audit-results/xg_source_schema.json"))
    parser.add_argument("--csv", type=Path, default=Path("audit-results/xg_source_schema.csv"))
    args = parser.parse_args()

    result = audit_sources(args.data_csv, args.epl_xlsm)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for source in result["sources"]:
        if source["xg_candidate_columns"]:
            rows.extend({"source": source["source"], **column} for column in source["xg_candidate_columns"])
        else:
            rows.append({
                "source": source["source"],
                "column": None,
                "column_index_0based": None,
                "non_null_rows": 0,
                "numeric_rows": 0,
                "non_null_rate": 0.0,
                "numeric_rate": 0.0,
            })
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.csv, index=False, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
