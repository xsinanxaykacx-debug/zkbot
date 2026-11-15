"""Build chronological features and evaluate real-data logistic baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from zk2.features.pipeline import build_features
from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel, FEATURES, TARGET_COLUMNS
from zk2.models.metrics import evaluate_probabilities


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "date_min": str(pd.to_datetime(frame["_date"]).min()),
        "date_max": str(pd.to_datetime(frame["_date"]).max()),
    }


def run_baselines(dataset_path: Path, features_path: Path, report_path: Path) -> dict:
    raw = pd.read_csv(dataset_path, compression={"method": "gzip", "mtime": 0}, low_memory=False)
    required = {"_date", "_team_home_canonical", "_team_away_canonical", "_league_canonical", "MSE", "MSD", *TARGET_COLUMNS.values()}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Locked dataset is missing feature-pipeline columns: {missing}")

    featured = build_features(raw)
    train, validation, test = chronological_split(featured)
    results = {}
    for market, target_column in TARGET_COLUMNS.items():
        model = BaselineModel(market)
        model.fit(train, train[target_column])
        class_order = tuple(model.classes_.tolist())
        market_result = {
            "class_order": list(class_order),
            "train_target_counts": {str(k): int(v) for k, v in train[target_column].value_counts().items()},
            "validation_target_counts": {str(k): int(v) for k, v in validation[target_column].value_counts().items()},
            "test_target_counts": {str(k): int(v) for k, v in test[target_column].value_counts().items()},
            "validation": evaluate_probabilities(
                validation[target_column].to_numpy(), model.predict_proba(validation), class_order=class_order
            ),
            "test": evaluate_probabilities(
                test[target_column].to_numpy(), model.predict_proba(test), class_order=class_order
            ),
        }
        results[market] = market_result

    export_columns = [
        "_source_dataset", "_source_row", "_date", "_season", "_league_canonical",
        "_team_home_canonical", "_team_away_canonical", "home_score", "away_score",
        "_y_1x2", "_y_ou25", "_y_kg", *FEATURES,
    ]
    missing_export = sorted(set(export_columns).difference(featured.columns))
    if missing_export:
        raise ValueError(f"Feature export is missing columns: {missing_export}")
    features_path.parent.mkdir(parents=True, exist_ok=True)
    featured.loc[:, export_columns].to_csv(features_path, index=False, compression="gzip")

    report = {
        "dataset_file": dataset_path.name,
        "dataset_sha256": sha256_file(dataset_path),
        "rows": int(len(featured)),
        "features_count": len(FEATURES),
        "feature_columns": list(FEATURES),
        "split": {
            "train": _partition_summary(train),
            "validation": _partition_summary(validation),
            "test": _partition_summary(test),
            "policy": "70/15/15 chronological split by distinct match dates; no shuffled folds",
        },
        "markets": results,
        "feature_file": features_path.name,
        "feature_file_sha256": sha256_file(features_path),
        "note": "These are first real-data logistic baselines, not tuned or calibrated models. Validation/test metrics are reported without performance thresholds or claims of profitability.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--features", type=Path, default=Path("zk2_features_v1.csv.gz"))
    parser.add_argument("--report", type=Path, default=Path("zk2_real_baseline_report.json"))
    args = parser.parse_args()
    report = run_baselines(args.dataset, args.features, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
