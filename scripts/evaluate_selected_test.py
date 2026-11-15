"""Evaluate validation-selected models on the locked test split exactly once.

This script must only be run after validation-only model selection is finalized.
It refits the selected model on train + validation, then scores the test partition.
No test result is used to change model selection or parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel, FEATURES, TARGET_COLUMNS
from zk2.models.lightgbm_baseline import LightGBMBaseline
from zk2.models.metrics import evaluate_probabilities


def _score(frame: pd.DataFrame, probabilities, target_column: str, class_order) -> dict:
    valid = frame[target_column].notna().to_numpy()
    y = frame.loc[valid, target_column].to_numpy()
    return evaluate_probabilities(y, probabilities[valid], class_order=class_order)


def run_selected_test(features_path: Path, search_report_path: Path, report_path: Path) -> dict:
    frame = pd.read_csv(features_path, compression="infer", low_memory=False)
    required = {"_date", *FEATURES, *TARGET_COLUMNS.values()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature artifact is missing required columns: {missing}")

    search = json.loads(search_report_path.read_text(encoding="utf-8"))
    train, validation, test = chronological_split(frame)
    fit_data = pd.concat([train, validation], ignore_index=True).sort_values("_date")

    report = {
        "method": (
            "One final holdout evaluation after validation-only selection. Selected models "
            "are refit on train + validation. Test is scored once and never used for tuning."
        ),
        "features_count": len(FEATURES),
        "fit_rows": int(len(fit_data)),
        "test_rows": int(len(test)),
        "fit_max_date": str(pd.to_datetime(fit_data["_date"]).max().date()),
        "test_min_date": str(pd.to_datetime(test["_date"]).min().date()),
        "test_max_date": str(pd.to_datetime(test["_date"]).max().date()),
        "markets": {},
    }

    for market, target_column in TARGET_COLUMNS.items():
        selected = search["markets"][market]
        class_order = ("H", "D", "A") if market == "1x2" else (0, 1)

        lr = BaselineModel(market).fit(fit_data, fit_data[target_column])
        lr_probabilities = lr.predict_proba(test)
        lr_metrics = _score(test, lr_probabilities, target_column, class_order)

        winner = selected["validation_winner"]
        market_result = {
            "validation_winner": winner,
            "validation_winner": winner,\n            "validation_best_candidate": selected["best_candidate"],\n            "validation_confirmation_bootstrap": selected["confirmation"]["paired_block_bootstrap"],
            "logistic_regression_test_reference": lr_metrics,
        }

        if winner == "lightgbm":
            candidate_id = int(selected["best_candidate"])
            candidates = selected["candidates_ranked_by_selection_log_loss"]
            candidate = next(item for item in candidates if int(item["candidate"]) == candidate_id)
            model = LightGBMBaseline(market, params=candidate["overrides"]).fit(
                fit_data, fit_data[target_column]
            )
            probabilities = model.predict_proba(test)
            selected_metrics = _score(test, probabilities, target_column, class_order)
            market_result["selected_model"] = "lightgbm"
            market_result["selected_parameters"] = candidate["overrides"]
            market_result["selected_test_metrics"] = selected_metrics
            market_result["selected_minus_lr_test_log_loss"] = (
                selected_metrics["log_loss"] - lr_metrics["log_loss"]
            )
        else:
            market_result["selected_model"] = "logistic_regression"
            market_result["selected_test_metrics"] = lr_metrics
            market_result["selected_minus_lr_test_log_loss"] = 0.0

        report["markets"][market] = market_result

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("search_report", type=Path)
    parser.add_argument("--report", type=Path, default=Path("zk2_selected_test_report.json"))
    args = parser.parse_args()
    print(json.dumps(run_selected_test(args.features, args.search_report, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
