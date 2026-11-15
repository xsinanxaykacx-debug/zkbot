"""Reusable source schema and data-quality checks."""

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


@dataclass
class ValidationReport:
    row_count: int
    missing_required_columns: list[str] = field(default_factory=list)
    column_fill_rates: dict[str, float] = field(default_factory=dict)
    invalid_date_count: int = 0
    duplicate_key_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.missing_required_columns


def validate_schema(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    inspect_columns: Iterable[str] = (),
    date_column: str | None = None,
    key_columns: Iterable[str] = (),
) -> ValidationReport:
    required = list(required_columns)
    inspected = list(dict.fromkeys([*required, *inspect_columns]))
    missing = [column for column in required if column not in frame.columns]
    rates = {
        column: float(frame[column].notna().mean())
        for column in inspected
        if column in frame.columns and len(frame) > 0
    }
    invalid_dates = 0
    if date_column and date_column in frame.columns:
        parsed = pd.to_datetime(frame[date_column], errors="coerce", dayfirst=True)
        invalid_dates = int(parsed.isna().sum())
    duplicate_count = 0
    keys = list(key_columns)
    if keys and all(column in frame.columns for column in keys):
        duplicate_count = int(frame.duplicated(keys, keep=False).sum())
    warnings = []
    if len(frame) == 0:
        warnings.append("Source contains zero rows.")
    if invalid_dates:
        warnings.append(f"{invalid_dates} rows have missing or unparseable dates.")
    if duplicate_count:
        warnings.append(f"{duplicate_count} rows participate in duplicate keys.")
    return ValidationReport(len(frame), missing, rates, invalid_dates, duplicate_count, warnings)
