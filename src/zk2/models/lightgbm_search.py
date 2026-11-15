"""Small, predeclared LightGBM search using validation only; never loads or scores test rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel, FEATURES, TARGET_COLUMNS
from zk2.models.lightgbm_baseline import DEFAULT_PARAMS, LightGBMBaseline
from zk2.models.metrics import evaluate_probabilities


# Six deliberately small configurations. These are fixed before looking at results.
SEARCH_GRID = (
    {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 7, "min_child_samples": 50, "reg_lambda": 5.0},
    {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 30, "reg_lambda": 2.0},
    {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 50, "reg_lambda": 5.0},
    {"n_estimators": 600, "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 80, "reg_lambda": 10.0},
    {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 80, "reg_lambda": 10.0},
    {"n_estimators": 800, "learning_rate": 0.01, "num_leaves": 7, "min_child_samples": 80, "reg_lambda": 10.0},
)


def _score(frame: pd.DataFrame, probabilities, target_column: str, class_order) -> dict:
    valid = frame[target_column].notna().to_numpy()
    y = frame.loc[valid, target_column].to_numpy()
    return evaluate_probabilities(y, probabilities[valid], class_order=class_order)


def run_validation_search(features_path: Path, report_path: Path) -> dict:
    frame = pd.read_csv(features_path, compression="infer", low_memory=False)
    required = {"_date", *FEATURES, *TARGET_COLUMNS.values()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature artifact is missing required columns: {missing}")

    train, validation, _ = chronological_split(frame)
    report = {
        "method": (
            "Six predeclared LightGBM configurations. Fit on chronological train only; "
            "rank on chronological validation only. Test partition is not read, scored, "
            "or used for selection."
        ),
        "features_count": len(FEATURES),
        "feature_columns": list(FEATURES),
        "grid_size": len(SEARCH_GRID),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_max_date": str(pd.to_datetime(train["_date"]).max().date()),
        "validation_min_date": str(pd.to_datetime(validation["_date"]).min().date()),
        "validation_max_date": str(pd.to_datetime(validation["_date"]).max().date()),
        "markets": {},
    }

    for market, target_column in TARGET_COLUMNS.items():
        class_order = ("H", "D", "A") if market == "1x2" else (0, 1)
        lr = BaselineModel(market).fit(train, train[target_column])
        lr_score = _score(validation, lr.predict_proba(validation), target_column, class_order)

        candidates = []
        for index, overrides in enumerate(SEARCH_GRID, start=1):
            params = {**DEFAULT_PARAMS, **overrides}
            model = LightGBMBaseline(market, params=params).fit(train, train[target_column])
            score = _score(
                validation, model.predict_proba(validation), target_column, class_order
            )
            candidates.append({
                "candidate": index,
                "overrides": overrides,
                "validation_metrics": score,
                "delta_log_loss_vs_lr": score["log_loss"] - lr_score["log_loss"],
            })

        candidates.sort(key=lambda item: item["validation_metrics"]["log_loss"])
        best = candidates[0]
        report["markets"][market] = {
            "class_order": list(class_order),
            "logistic_regression_validation": lr_score,
            "candidates_ranked_by_validation_log_loss": candidates,
            "best_candidate": best["candidate"],
            "best_lightgbm_validation": best["validation_metrics"],
            "best_delta_log_loss_vs_lr": best["delta_log_loss_vs_lr"],
            "validation_winner": (
                "lightgbm" if best["delta_log_loss_vs_lr"] < 0 else "logistic_regression"
            ),
            "test_evaluated": False,
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--report", type=Path, default=Path("zk2_lightgbm_search_report.json"))
    args = parser.parse_args()
    print(json.dumps(run_validation_search(args.features, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--report", type=Path, default=Path("zk2_lightgbm_search_report.json"))
    args = parser.parse_args()
    print(json.dumps(run_validation_search(args.features, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
