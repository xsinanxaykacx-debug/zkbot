"""Validation-only LightGBM search with an independent chronological confirmation slice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel, FEATURES, TARGET_COLUMNS
from zk2.models.lightgbm_baseline import DEFAULT_PARAMS, LightGBMBaseline
from zk2.models.metrics import evaluate_probabilities


# Six deliberately small configurations, fixed before looking at results.
SEARCH_GRID = (
    {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 7, "min_child_samples": 50, "reg_lambda": 5.0},
    {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 30, "reg_lambda": 2.0},
    {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 50, "reg_lambda": 5.0},
    {"n_estimators": 600, "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 80, "reg_lambda": 10.0},
    {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 80, "reg_lambda": 10.0},
    {"n_estimators": 800, "learning_rate": 0.01, "num_leaves": 7, "min_child_samples": 80, "reg_lambda": 10.0},
)

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_DATES = 5
BOOTSTRAP_SEED = 20261115


def _valid_rows(frame: pd.DataFrame, target_column: str):
    mask = frame[target_column].notna().to_numpy()
    return mask, frame.loc[mask, target_column].to_numpy()


def _score(frame: pd.DataFrame, probabilities, target_column: str, class_order) -> dict:
    mask, y = _valid_rows(frame, target_column)
    return evaluate_probabilities(y, probabilities[mask], class_order=class_order)


def _row_log_losses(y_true, probabilities, class_order) -> np.ndarray:
    class_to_index = {label: index for index, label in enumerate(class_order)}
    encoded = np.asarray([class_to_index[label] for label in np.asarray(y_true).tolist()], dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    chosen = probs[np.arange(len(encoded)), encoded]
    return -np.log(np.clip(chosen, 1e-15, 1.0))


def paired_date_block_bootstrap(
    frame: pd.DataFrame,
    target_column: str,
    lr_probabilities,
    lgbm_probabilities,
    class_order,
    *,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    block_dates: int = BOOTSTRAP_BLOCK_DATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap paired per-match loss differences in moving blocks of match dates.

    Positive difference means LR loss minus LightGBM loss is positive, so LightGBM
    performs better. All matches on a sampled date stay together.
    """
    mask, y = _valid_rows(frame, target_column)
    eval_frame = frame.loc[mask].copy()
    if eval_frame.empty:
        raise ValueError(f"No non-missing targets available for {target_column}.")

    lr_loss = _row_log_losses(y, np.asarray(lr_probabilities)[mask], class_order)
    lgbm_loss = _row_log_losses(y, np.asarray(lgbm_probabilities)[mask], class_order)
    paired_difference = lr_loss - lgbm_loss

    dates = pd.to_datetime(eval_frame["_date"]).to_numpy()
    date_codes, unique_dates = pd.factorize(dates, sort=True)
    n_dates = len(unique_dates)
    if n_dates < 2:
        raise ValueError("At least two distinct confirmation dates are required.")

    rows_by_date = [np.flatnonzero(date_codes == code) for code in range(n_dates)]
    block_length = min(max(1, int(block_dates)), n_dates)
    blocks_needed = int(np.ceil(n_dates / block_length))
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstrap, dtype=float)

    for replicate in range(n_bootstrap):
        starts = rng.integers(0, n_dates, size=blocks_needed)
        sampled_dates = np.concatenate([
            (start + np.arange(block_length)) % n_dates for start in starts
        ])[:n_dates]
        sampled_rows = np.concatenate([rows_by_date[index] for index in sampled_dates])
        bootstrap_means[replicate] = float(np.mean(paired_difference[sampled_rows]))

    observed = float(np.mean(paired_difference))
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "method": "paired moving-block bootstrap over chronological match dates",
        "replicates": int(n_bootstrap),
        "block_length_distinct_match_dates": int(block_length),
        "distinct_match_dates": int(n_dates),
        "n_matches": int(len(y)),
        "observed_lr_minus_lightgbm_log_loss": observed,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "lightgbm_supported_at_95_percent": bool(lower > 0.0),
        "interpretation": (
            "Positive values favor LightGBM. LightGBM is selected only when the full "
            "95% interval is above zero; otherwise retain logistic regression."
        ),
    }


def _split_validation_by_date(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = pd.Index(pd.to_datetime(validation["_date"]).drop_duplicates().sort_values())
    if len(unique_dates) < 4:
        raise ValueError("At least four distinct validation dates are required.")
    midpoint = len(unique_dates) // 2
    selection_dates = unique_dates[:midpoint]
    confirmation_dates = unique_dates[midpoint:]
    selection = validation.loc[pd.to_datetime(validation["_date"]).isin(selection_dates)].copy()
    confirmation = validation.loc[pd.to_datetime(validation["_date"]).isin(confirmation_dates)].copy()
    if selection.empty or confirmation.empty:
        raise ValueError("Validation selection/confirmation split produced an empty partition.")
    return selection, confirmation


def run_validation_search(features_path: Path, report_path: Path) -> dict:
    frame = pd.read_csv(features_path, compression="infer", low_memory=False)
    required = {"_date", *FEATURES, *TARGET_COLUMNS.values()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature artifact is missing required columns: {missing}")

    train, validation, _ = chronological_split(frame)
    selection, confirmation = _split_validation_by_date(validation)
    report = {
        "method": (
            "Six predeclared LightGBM configurations ranked on the early half of the "
            "chronological validation period. The selected candidate is compared with LR "
            "on the later, untouched validation confirmation half using paired date-block "
            "bootstrap. Test partition is never scored or used for selection."
        ),
        "features_count": len(FEATURES),
        "feature_columns": list(FEATURES),
        "grid_size": len(SEARCH_GRID),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "selection_rows": int(len(selection)),
        "confirmation_rows": int(len(confirmation)),
        "train_max_date": str(pd.to_datetime(train["_date"]).max().date()),
        "selection_min_date": str(pd.to_datetime(selection["_date"]).min().date()),
        "selection_max_date": str(pd.to_datetime(selection["_date"]).max().date()),
        "confirmation_min_date": str(pd.to_datetime(confirmation["_date"]).min().date()),
        "confirmation_max_date": str(pd.to_datetime(confirmation["_date"]).max().date()),
        "markets": {},
    }

    for market, target_column in TARGET_COLUMNS.items():
        class_order = ("H", "D", "A") if market == "1x2" else (0, 1)
        lr = BaselineModel(market).fit(train, train[target_column])
        lr_selection_probabilities = lr.predict_proba(selection)
        lr_selection_score = _score(
            selection, lr_selection_probabilities, target_column, class_order
        )

        candidates = []
        candidate_models = {}
        for index, overrides in enumerate(SEARCH_GRID, start=1):
            params = {**DEFAULT_PARAMS, **overrides}
            model = LightGBMBaseline(market, params=params).fit(train, train[target_column])
            probabilities = model.predict_proba(selection)
            score = _score(selection, probabilities, target_column, class_order)
            candidates.append({
                "candidate": index,
                "overrides": overrides,
                "selection_metrics": score,
                "delta_log_loss_vs_lr": score["log_loss"] - lr_selection_score["log_loss"],
            })
            candidate_models[index] = model

        candidates.sort(key=lambda item: item["selection_metrics"]["log_loss"])
        best = candidates[0]
        best_model = candidate_models[best["candidate"]]

        lr_confirmation_probabilities = lr.predict_proba(confirmation)
        best_confirmation_probabilities = best_model.predict_proba(confirmation)
        lr_confirmation_score = _score(
            confirmation, lr_confirmation_probabilities, target_column, class_order
        )
        best_confirmation_score = _score(
            confirmation, best_confirmation_probabilities, target_column, class_order
        )
        bootstrap = paired_date_block_bootstrap(
            confirmation,
            target_column,
            lr_confirmation_probabilities,
            best_confirmation_probabilities,
            class_order,
        )
        selected_winner = (
            "lightgbm" if bootstrap["lightgbm_supported_at_95_percent"]
            else "logistic_regression"
        )

        report["markets"][market] = {
            "class_order": list(class_order),
            "logistic_regression_selection": lr_selection_score,
            "candidates_ranked_by_selection_log_loss": candidates,
            "best_candidate": best["candidate"],
            "best_lightgbm_selection": best["selection_metrics"],
            "best_delta_log_loss_vs_lr_selection": best["delta_log_loss_vs_lr"],
            "confirmation": {
                "logistic_regression": lr_confirmation_score,
                "selected_lightgbm_candidate": best_confirmation_score,
                "paired_block_bootstrap": bootstrap,
            },
            "validation_winner": selected_winner,
            "selection_policy": (
                "LightGBM only if the 95% paired date-block bootstrap CI for "
                "LR loss minus LightGBM loss is entirely above zero; otherwise LR."
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
