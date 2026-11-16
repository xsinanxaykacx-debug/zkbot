#!/usr/bin/env python3
"""Build a chronological 70/15/15 split manifest for xg18.

Rows with invalid/missing dates are quarantined. All matches on a calendar
date stay in the same partition. This script does not train or score models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SOURCE = Path("xg18.csv")
OUT = Path("split-results")
DATASET_ID = "zk2-xg18-chronological-split-v1"
RATIOS = (0.70, 0.15, 0.15)


def nearest_boundary(cumulative: list[int], target: float, minimum: int = 1) -> int:
    """Return a group boundary index (number of date groups on the left)."""
    candidates = range(minimum, len(cumulative))
    return min(candidates, key=lambda i: (abs(cumulative[i - 1] - target), i))


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Required source file not found: {SOURCE}")

    raw_bytes = SOURCE.read_bytes()
    df = pd.read_csv(SOURCE, encoding="utf-8-sig", low_memory=False)
    required = {"Date", "League", "HomeTeam", "AwayTeam"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"xg18.csv missing required columns: {missing}")

    dates = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    valid = df.loc[dates.notna()].copy()
    valid["_split_date"] = dates.loc[dates.notna()]
    quarantine = df.loc[dates.isna()].copy()

    if valid.empty:
        raise ValueError("No rows have a valid match date.")

    # Date groups are indivisible: no calendar date can cross a split boundary.
    per_date = valid.groupby("_split_date", sort=True).size()
    date_counts = [int(n) for n in per_date.tolist()]
    cumulative = []
    running = 0
    for n in date_counts:
        running += n
        cumulative.append(running)

    n = len(valid)
    train_target = n * RATIOS[0]
    first_boundary = nearest_boundary(cumulative, train_target)
    first_count = cumulative[first_boundary - 1]

    confirmation_target = n * (RATIOS[0] + RATIOS[1])
    second_boundary = nearest_boundary(cumulative, confirmation_target, first_boundary + 1)
    second_count = cumulative[second_boundary - 1]

    train_end = per_date.index[first_boundary - 1]
    confirmation_end = per_date.index[second_boundary - 1]
    train_start = per_date.index[0]
    confirmation_start = per_date.index[first_boundary]
    test_start = per_date.index[second_boundary]
    test_end = per_date.index[-1]

    assignments = pd.DataFrame({
        "source_row_index": df.index,
        "date": dates.dt.strftime("%Y-%m-%d"),
        "partition": "quarantine_missing_date",
    })
    assignments.loc[valid.index[valid["_split_date"] <= train_end], "partition"] = "train"
    assignments.loc[valid.index[(valid["_split_date"] > train_end) & (valid["_split_date"] <= confirmation_end)], "partition"] = "confirmation"
    assignments.loc[valid.index[valid["_split_date"] > confirmation_end], "partition"] = "test"

    counts = assignments["partition"].value_counts().to_dict()
    expected = {
        "train": first_count,
        "confirmation": second_count - first_count,
        "test": n - second_count,
        "quarantine_missing_date": len(quarantine),
    }
    for key, value in expected.items():
        if int(counts.get(key, 0)) != value:
            raise AssertionError(f"Partition count mismatch for {key}: {counts.get(key, 0)} != {value}")
    if sum(expected[k] for k in ("train", "confirmation", "test")) != n:
        raise AssertionError("Dated row partition counts do not sum to the dated source rows.")

    # Explicitly prove that no date appears in more than one chronological split.
    date_partition_counts = assignments[assignments["date"].notna()].groupby("date")["partition"].nunique()
    if (date_partition_counts > 1).any():
        raise AssertionError("At least one calendar date crosses partition boundaries.")

    OUT.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(OUT / "xg18_split_assignments.csv", index=False)
    manifest = {
        "dataset_id": DATASET_ID,
        "source_file": str(SOURCE),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_rows": int(len(df)),
        "dated_rows": int(n),
        "missing_or_invalid_date_rows_quarantined": int(len(quarantine)),
        "ratios_requested": {"train": 0.70, "confirmation": 0.15, "test": 0.15},
        "target_counts_approximate": {
            "train": round(n * 0.70),
            "confirmation": round(n * 0.15),
            "test": n - round(n * 0.70) - round(n * 0.15),
        },
        "actual_counts": expected,
        "date_ranges": {
            "train": {"start": str(train_start.date()), "end": str(train_end.date())},
            "confirmation": {"start": str(confirmation_start.date()), "end": str(confirmation_end.date())},
            "test": {"start": str(test_start.date()), "end": str(test_end.date())},
        },
        "distinct_dates": int(len(per_date)),
        "date_tie_policy": "All rows with the same normalized calendar date are assigned to one partition.",
        "boundary_policy": "Chronological date-group boundaries selected nearest to cumulative 70% and 85% row targets; exact ratios may differ slightly because dates are indivisible.",
        "quarantine_policy": "Rows with missing or invalid dates are excluded from train, confirmation, and test and retained only in the quarantine partition.",
        "row_identity": "Zero-based source CSV row index (header excluded from the index).",
        "training_or_scoring_performed": False,
        "final_test_policy": "Test partition is sealed; no tuning, fitting, or score calculation is performed by this workflow.",
    }
    (OUT / "xg18_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
