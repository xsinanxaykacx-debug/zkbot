#!/usr/bin/env python3
"""Prepare the one-time XG18 V3 final evaluation inputs.

The official train and validation partitions are combined for refitting only
after all model/threshold decisions are frozen. The final test is kept separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ABANDONED = {
    "4203250": ("Reading", "Port Vale", 0, 0),
    "4203025": ("Cambridge United", "Bolton Wanderers", 0, 0),
}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def attach(base: pd.DataFrame, odds: pd.DataFrame, label: str) -> pd.DataFrame:
    if base["MatchID"].astype(str).duplicated().any():
        raise ValueError(f"{label}: duplicate MatchID")
    lookup = odds[["MatchID", *[c for c in odds if c.startswith("odds_")]]].copy()
    lookup["MatchID"] = lookup["MatchID"].astype(str)
    if lookup["MatchID"].duplicated().any():
        raise ValueError("Merged odds MatchID is not unique")
    out = base.copy()
    out["MatchID"] = out["MatchID"].astype(str)
    out = out.merge(lookup, on="MatchID", how="left", validate="one_to_one", sort=False)
    if len(out) != len(base):
        raise RuntimeError(f"{label}: row count changed during odds join")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--merged-odds", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()

    train, validation, test, odds = map(read, (a.train, a.validation, a.test, a.merged_odds))
    id_sets = [set(d["MatchID"].astype(str)) for d in (train, validation, test)]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Official train/validation/test MatchIDs overlap")
    if len(set.union(*id_sets)) != sum(map(len, id_sets)):
        raise ValueError("Official partition ID reconciliation failed")

    trainval = pd.concat([train, validation], ignore_index=True)
    trainval = attach(trainval, odds, "train+validation")
    final_test = attach(test, odds, "final_test")

    ids = trainval["MatchID"].astype(str)
    abandoned = trainval.loc[ids.isin(ABANDONED)].copy()
    if set(abandoned["MatchID"].astype(str)) != set(ABANDONED):
        raise ValueError("Expected both abandoned 0-0 fixtures in training+validation")
    for row in abandoned.itertuples():
        home, away, hg, ag = ABANDONED[str(row.MatchID)]
        if (str(row.HomeTeam), str(row.AwayTeam), int(row.FTHG), int(row.FTAG)) != (home, away, hg, ag):
            raise ValueError(f"Abandoned fixture identity/score mismatch: {row.MatchID}")
    clean_trainval = trainval.loc[~ids.isin(ABANDONED)].copy()

    if set(clean_trainval["MatchID"].astype(str)) & set(final_test["MatchID"].astype(str)):
        raise ValueError("Final-test MatchID leaked into fitting data")
    if pd.to_datetime(clean_trainval["Date"], utc=True).max() >= pd.to_datetime(final_test["Date"], utc=True).min():
        raise ValueError("Training+validation is not strictly earlier than final test")

    a.outdir.mkdir(parents=True, exist_ok=True)
    clean_trainval.to_csv(a.outdir / "xg18_final_fit_train.csv", index=False)
    final_test.to_csv(a.outdir / "xg18_final_test.csv", index=False)
    report = {
        "policy": "one-time final evaluation; no test outcomes used for fitting, tuning, weights, or calibration",
        "official_train_rows": len(train),
        "official_validation_rows": len(validation),
        "fit_rows_before_quarantine": len(trainval),
        "abandoned_rows_quarantined_from_fit": len(abandoned),
        "fit_rows_after_quarantine": len(clean_trainval),
        "final_test_rows": len(final_test),
        "final_test_ids_used_for_fit": 0,
        "odds_csv_files_policy": "only the existing 70 repository CSV files; no external acquisition",
        "missing_odds_policy": "missing odds remain null; no imputation or fabrication",
        "quarantined_match_ids": sorted(ABANDONED),
        "final_test_date_range": {
            "start": str(pd.to_datetime(final_test["Date"], utc=True).min()),
            "end": str(pd.to_datetime(final_test["Date"], utc=True).max()),
        },
    }
    (a.outdir / "final_eval_input_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
