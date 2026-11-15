"""Compare fixed LightGBM baselines with logistic regression on identical chronological rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel, FEATURES, TARGET_COLUMNS
from zk2.models.lightgbm_baseline import LightGBMBaseline, DEFAULT_PARAMS
from zk2.models.metrics import evaluate_probabilities


def _evaluate_same_rows(frame: pd.DataFrame, probabilities, target_column: str, class_order):
    valid = frame[target_column].notna().to_numpy()
    y = frame.loc[valid, target_column].to_numpy()
    return evaluate_probabilities(y, probabilities[valid], class_order=class_order)


def run_lightgbm_benchmark(features_path: Path, report_path: Path) -> dict:
    frame = pd.read_csv(features_path, compression="infer", low_memory=False)
    required = {"_date", *FEATURES, *TARGET_COLUMNS.values()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature artifact is missing required columns: {missing}")

    train, validation, test = chronological_split(frame)
    report = {
        "method": (
            "Fixed LightGBM parameters declared before test evaluation; same 32 features, "
            "same chronological 70/15/15 split, same valid-target rows as logistic regression. "
            "No test-driven tuning."
        ),
        "features_count": len(FEATURES),
        "feature_columns": list(FEATURES),
        "parameters": DEFAULT_PARAMS,
        "split": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "train_max_date": str(pd.to_datetime(train["_date"]).max().date()),
            "validation_min_date": str(pd.to_datetime(validation["_date"]).min().date()),
            "validation_max_date": str(pd.to_datetime(validation["_date"]).max().date()),
            "test_min_date": str(pd.to_datetime(test["_date"]).min().date()),
            "test_max_date": str(pd.to_datetime(test["_date"]).max().date()),
        },
        "markets": {},
    }

    for market, target_column in TARGET_COLUMNS.items():
        class_order = tuple(
            ("H", "D", "A") if market == "1x2" else (0, 1)
        )
        lr = BaselineModel(market).fit(train, train[target_column])
        lgbm = LightGBMBaseline(market).fit(train, train[target_column])

        lr_validation_probabilities = lr.predict_proba(validation)
        lgbm_validation_probabilities = lgbm.predict_proba(validation)
        lr_test_probabilities = lr.predict_proba(test)
        lgbm_test_probabilities = lgbm.predict_proba(test)

        lr_validation = _evaluate_same_rows(
            validation, lr_validation_probabilities, target_column, class_order
        )
        lgbm_validation = _evaluate_same_rows(
            validation, lgbm_validation_probabilities, target_column, class_order
        )
        lr_test = _evaluate_same_rows(test, lr_test_probabilities, target_column, class_order)
        lgbm_test = _evaluate_same_rows(test, lgbm_test_probabilities, target_column, class_order)

        report["markets"][market] = {
            "class_order": list(class_order),
            "validation": {
                "logistic_regression": lr_validation,
                "lightgbm": lgbm_validation,
                "lightgbm_minus_logistic_log_loss": (
                    lgbm_validation["log_loss"] - lr_validation["log_loss"]
                ),
            },
            "test": {
                "logistic_regression": lr_test,
                "lightgbm": lgbm_test,
                "lightgbm_minus_logistic_log_loss": (
                    lgbm_test["log_loss"] - lr_test["log_loss"]
                ),
            },
            "same_validation_rows": lr_validation["n"] == lgbm_validation["n"],
            "same_test_rows": lr_test["n"] == lgbm_test["n"],
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--report", type=Path, default=Path("zk2_lightgbm_report.json"))
    args = parser.parse_args()
    print(json.dumps(run_lightgbm_benchmark(args.features, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
