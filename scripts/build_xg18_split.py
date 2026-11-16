#!/usr/bin/env python3
"""Create and validate the versioned XG18 chronological 70/15/15 split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SOURCE = Path("xg18.csv")
OUT = Path("split-results")
DATASET_ID = "zk2-xg18-chronological-split-v1"


def nearest_boundary(cumulative: list[int], target: float, minimum: int = 1) -> int:
    candidates = range(minimum, len(cumulative))
    if not candidates:
        raise ValueError("Not enough distinct match dates to create all partitions.")
    return min(candidates, key=lambda i: (abs(cumulative[i - 1] - target), i))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Required source file not found: {SOURCE}")

    raw_bytes = SOURCE.read_bytes()
    df = pd.read_csv(SOURCE, encoding="utf-8-sig", low_memory=False)
    required = {"Date", "League", "HomeTeam", "AwayTeam"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"xg18.csv missing required columns: {missing}")

    parsed_dates = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    valid_mask = parsed_dates.notna()
    dated = df.loc[valid_mask].copy()
    dated["_split_date"] = parsed_dates.loc[valid_mask]
    quarantine = df.loc[~valid_mask].copy()

    if len(df) != 18509 or len(dated) != 18505 or len(quarantine) != 4:
        raise AssertionError(
            f"Source row-count drift: total={len(df)}, dated={len(dated)}, "
            f"quarantine={len(quarantine)}; expected 18509/18505/4."
        )

    # Stable chronological ordering; calendar-day groups are indivisible.
    dated = dated.sort_values("_split_date", kind="mergesort")
    per_date = dated.groupby("_split_date", sort=True).size()
    cumulative: list[int] = []
    running = 0
    for group_size in per_date.tolist():
        running += int(group_size)
        cumulative.append(running)

    n = len(dated)
    first_cut = nearest_boundary(cumulative, n * 0.70)
    second_cut = nearest_boundary(cumulative, n * 0.85, first_cut + 1)

    first_date = per_date.index[first_cut - 1]
    second_date = per_date.index[second_cut - 1]
    train = dated.loc[dated["_split_date"] <= first_date].drop(columns="_split_date")
    val = dated.loc[(dated["_split_date"] > first_date) & (dated["_split_date"] <= second_date)].drop(columns="_split_date")
    test = dated.loc[dated["_split_date"] > second_date].drop(columns="_split_date")

    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "xg18_train.csv": train,
        "xg18_val.csv": val,
        "xg18_test.csv": test,
        "xg18_quarantine.csv": quarantine,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUT / filename, index=False, encoding="utf-8")

    # Re-read only non-test outputs for integrity checks. Never inspect test rows.
    train_check = pd.read_csv(OUT / "xg18_train.csv", usecols=["Date"])
    val_check = pd.read_csv(OUT / "xg18_val.csv", usecols=["Date"])
    quarantine_check = pd.read_csv(OUT / "xg18_quarantine.csv")
    if len(train_check) != len(train) or len(val_check) != len(val) or len(quarantine_check) != 4:
        raise AssertionError("Saved CSV row-count verification failed.")
    if len(train) + len(val) + len(test) != 18505:
        raise AssertionError("Dated partition counts do not sum to 18,505.")
    if len(train) + len(val) + len(test) + len(quarantine) != len(df):
        raise AssertionError("All partition and quarantine counts do not sum to source rows.")

    train_dates = pd.to_datetime(train_check["Date"], errors="coerce", utc=True)
    val_dates = pd.to_datetime(val_check["Date"], errors="coerce", utc=True)
    if train_dates.isna().any() or val_dates.isna().any():
        raise AssertionError("A missing/invalid date entered train or validation.")
    if not train_dates.is_monotonic_increasing or not val_dates.is_monotonic_increasing:
        raise AssertionError("Train or validation output is not chronologically sorted.")
    if train_dates.max() >= val_dates.min():
        raise AssertionError("Train and validation date ranges overlap or are out of order.")
    if val_dates.max() >= pd.Timestamp(second_date, tz="UTC"):
        # Validation's last date is expected to be exactly the second cut date.
        if val_dates.max().normalize() != pd.Timestamp(second_date, tz="UTC"):
            raise AssertionError("Validation end date does not match the selected date boundary.")

    train_start = pd.to_datetime(train["Date"], utc=True).min().date().isoformat()
    train_end = pd.to_datetime(train["Date"], utc=True).max().date().isoformat()
    val_start = pd.to_datetime(val["Date"], utc=True).min().date().isoformat()
    val_end = pd.to_datetime(val["Date"], utc=True).max().date().isoformat()

    # Test dates are read from the in-memory partition only to record boundaries;
    # no test rows, features, labels, or outcomes are inspected or printed.
    test_start = str(pd.to_datetime(dated.loc[dated["_split_date"] > second_date, "_split_date"].min()).date())
    test_end = str(pd.to_datetime(dated["_split_date"].max()).date())

    manifest = {
        "dataset_id": DATASET_ID,
        "source_file": str(SOURCE),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_rows": len(df),
        "dated_rows": len(dated),
        "quarantine_rows": len(quarantine),
        "requested_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "target_counts": {"train": 12956, "validation": 2776, "test": 2773},
        "actual_counts": {"train": len(train), "validation": len(val), "test": len(test), "quarantine": len(quarantine)},
        "date_ranges": {
            "train": {"start": train_start, "end": train_end},
            "validation": {"start": val_start, "end": val_end},
            "test": {"start": test_start, "end": test_end},
        },
        "same_day_policy": "All matches on the same normalized calendar date stay in one partition.",
        "boundary_policy": "Chronological date groups are assigned to the nearest cumulative 70% and 85% row targets; actual counts may vary slightly because date groups are indivisible.",
        "quarantine_policy": "All four missing/invalid-date rows are stored separately and excluded from all three partitions.",
        "test_policy": "SEALED. No test-row inspection, model training, tuning, or evaluation is performed. Test CSV is published as a separate restricted-purpose artifact.",
        "test_sha256": sha256_file(OUT / "xg18_test.csv"),
        "checks": {
            "source_total_18509": True,
            "dated_total_18505": True,
            "quarantine_exactly_4": True,
            "dated_partitions_sum_18505": True,
            "train_before_validation": True,
            "same_day_groups_indivisible": True,
            "chronological_sort_train_validation": True,
        },
    }
    (OUT / "xg18_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "dataset_id": DATASET_ID,
        "actual_counts": manifest["actual_counts"],
        "date_ranges": manifest["date_ranges"],
        "checks": manifest["checks"],
        "test_policy": "SEALED; no test rows or metrics displayed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
