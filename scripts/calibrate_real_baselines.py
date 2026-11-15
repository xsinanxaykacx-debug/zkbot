"""Compare Platt and isotonic calibration without validation/test leakage.

The validation period is split chronologically: the early portion fits
calibrators and the later portion selects the method. The selected method is
then refit on all validation predictions and evaluated once on the locked test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel, TARGET_COLUMNS
from zk2.models.calibration import OneVsRestCalibrator
from zk2.models.metrics import evaluate_probabilities


def _valid_arrays(frame: pd.DataFrame, probabilities: np.ndarray, target: str):
    valid = frame[target].notna().to_numpy()
    return frame.loc[valid, target].to_numpy(), probabilities[valid]


def _chronological_calibration_split(validation: pd.DataFrame):
    dates = np.sort(pd.to_datetime(validation["_date"]).dropna().unique())
    if len(dates) < 3:
        raise ValueError("Validation needs at least three distinct dates for calibration selection.")
    cut = int(np.floor(len(dates) * 0.70))
    cut = min(max(cut, 1), len(dates) - 1)
    fit_dates = set(dates[:cut])
    fit_mask = pd.to_datetime(validation["_date"]).isin(fit_dates)
    fit = validation.loc[fit_mask]
    selection = validation.loc[~fit_mask]
    if fit.empty or selection.empty:
        raise ValueError("Chronological calibration fit/selection split is empty.")
    return fit, selection


def run_calibration(features_path: Path, report_path: Path) -> dict:
    frame = pd.read_csv(features_path, compression="infer", low_memory=False)
    train, validation, test = chronological_split(frame)
    calibration_fit, calibration_selection = _chronological_calibration_split(validation)
    report = {
        "method": "Chronological validation calibration-fit/selection split (70/30 by distinct dates); selected calibrator refit on all validation; locked test evaluated once.",
        "validation_fit": {
            "rows": int(len(calibration_fit)),
            "date_min": str(pd.to_datetime(calibration_fit["_date"]).min().date()),
            "date_max": str(pd.to_datetime(calibration_fit["_date"]).max().date()),
        },
        "validation_selection": {
            "rows": int(len(calibration_selection)),
            "date_min": str(pd.to_datetime(calibration_selection["_date"]).min().date()),
            "date_max": str(pd.to_datetime(calibration_selection["_date"]).max().date()),
        },
        "test": {
            "rows": int(len(test)),
            "date_min": str(pd.to_datetime(test["_date"]).min().date()),
            "date_max": str(pd.to_datetime(test["_date"]).max().date()),
        },
        "markets": {},
    }

    for market, target_column in TARGET_COLUMNS.items():
        model = BaselineModel(market)
        model.fit(train, train[target_column])
        class_order = tuple(model.classes_.tolist())
        p_fit = model.predict_proba(calibration_fit)
        p_selection = model.predict_proba(calibration_selection)
        y_fit, p_fit = _valid_arrays(calibration_fit, p_fit, target_column)
        y_selection, p_selection = _valid_arrays(calibration_selection, p_selection, target_column)
        if set(y_fit.tolist()) != set(class_order) or set(y_selection.tolist()) != set(class_order):
            raise ValueError(f"{market}: each validation calibration partition must contain every class.")

        selection_result = {
            "baseline": evaluate_probabilities(y_selection, p_selection, class_order=class_order),
            "calibrators": {},
        }
        fitted_calibrators = {}
        for method in ("platt", "isotonic"):
            calibrator = OneVsRestCalibrator(method).fit(p_fit, y_fit, class_order)
            calibrated_selection = calibrator.predict_proba(p_selection)
            selection_result["calibrators"][method] = evaluate_probabilities(
                y_selection, calibrated_selection, class_order=class_order
            )
            fitted_calibrators[method] = calibrator

        winner = min(
            selection_result["calibrators"],
            key=lambda method: selection_result["calibrators"][method]["log_loss"],
        )
        # Refit the selected calibrator on all validation predictions.
        p_validation = model.predict_proba(validation)
        y_validation, p_validation = _valid_arrays(validation, p_validation, target_column)
        final_calibrator = OneVsRestCalibrator(winner).fit(
            p_validation, y_validation, class_order
        )
        p_test = model.predict_proba(test)
        y_test, p_test = _valid_arrays(test, p_test, target_column)
        calibrated_test = final_calibrator.predict_proba(p_test)

        report["markets"][market] = {
            "class_order": list(class_order),
            "validation_selection": selection_result,
            "selected_method": winner,
            "test_baseline": evaluate_probabilities(y_test, p_test, class_order=class_order),
            "test_calibrated": evaluate_probabilities(
                y_test, calibrated_test, class_order=class_order
            ),
            "test_log_loss_delta": round(
                evaluate_probabilities(y_test, calibrated_test, class_order=class_order)["log_loss"]
                - evaluate_probabilities(y_test, p_test, class_order=class_order)["log_loss"],
                8,
            ),
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--report", type=Path, default=Path("zk2_calibration_report.json"))
    args = parser.parse_args()
    print(json.dumps(run_calibration(args.features, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
