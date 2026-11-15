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


def _paired_logloss_delta_ci(y_true, baseline_probs, candidate_probs, class_order, *, seed=20261115, n_boot=2000):
    """Paired bootstrap CI for candidate minus baseline per-row log loss."""
    class_to_index = {label: index for index, label in enumerate(class_order)}
    y_index = np.asarray([class_to_index[label] for label in np.asarray(y_true).tolist()])
    base = np.clip(np.asarray(baseline_probs, dtype=float), 1e-15, 1.0)
    candidate = np.clip(np.asarray(candidate_probs, dtype=float), 1e-15, 1.0)
    rows = np.arange(len(y_index))
    delta = np.log(candidate[rows, y_index]) * -1 - np.log(base[rows, y_index]) * -1
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[sampled].mean(axis=1)
    return {
        "mean_delta_candidate_minus_baseline": float(delta.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "bootstrap_samples": n_boot,
    }


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
        candidate_probabilities = {}
        for method in ("platt", "isotonic"):
            calibrator = OneVsRestCalibrator(method).fit(p_fit, y_fit, class_order)
            calibrated_selection = calibrator.predict_proba(p_selection)
            selection_result["calibrators"][method] = evaluate_probabilities(
                y_selection, calibrated_selection, class_order=class_order
            )
            candidate_probabilities[method] = calibrated_selection

        best_method = min(
            selection_result["calibrators"],
            key=lambda method: selection_result["calibrators"][method]["log_loss"],
        )
        improvement_evidence = _paired_logloss_delta_ci(
            y_selection, p_selection, candidate_probabilities[best_method], class_order
        )
        # Keep the uncalibrated baseline unless the paired 95% CI supports a
        # genuine improvement. This prevents selecting a tiny noisy gain.
        winner = best_method if improvement_evidence["ci95_upper"] < 0 else "none"

        p_validation = model.predict_proba(validation)
        y_validation, p_validation = _valid_arrays(validation, p_validation, target_column)
        p_test = model.predict_proba(test)
        y_test, p_test = _valid_arrays(test, p_test, target_column)
        baseline_test_metrics = evaluate_probabilities(y_test, p_test, class_order=class_order)

        if winner == "none":
            calibrated_test = p_test
        else:
            final_calibrator = OneVsRestCalibrator(winner).fit(
                p_validation, y_validation, class_order
            )
            calibrated_test = final_calibrator.predict_proba(p_test)

        calibrated_test_metrics = evaluate_probabilities(
            y_test, calibrated_test, class_order=class_order
        )
        report["markets"][market] = {
            "class_order": list(class_order),
            "validation_selection": selection_result,
            "best_candidate": best_method,
            "best_candidate_paired_bootstrap": improvement_evidence,
            "selected_method": winner,
            "test_baseline": baseline_test_metrics,
            "test_calibrated": calibrated_test_metrics,
            "test_log_loss_delta": round(
                calibrated_test_metrics["log_loss"] - baseline_test_metrics["log_loss"], 8
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
