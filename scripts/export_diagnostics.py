"""Export row-level predictions for the fixed LR confirmation and final-test models.

This is a diagnostic reconstruction of the already selected LR models. It does not
search parameters or select models. Confirmation predictions use train-only fits;
final-test predictions use train+validation fits, matching the original evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel, FEATURES, TARGET_COLUMNS
from zk2.models.lightgbm_search import _split_validation_by_date
from zk2.models.metrics import evaluate_probabilities


def _probability_columns(market: str, probabilities: np.ndarray) -> dict[str, np.ndarray]:
    if market == "1x2":
        return {
            "p_H": probabilities[:, 0],
            "p_D": probabilities[:, 1],
            "p_A": probabilities[:, 2],
        }
    if market == "ou25":
        return {"p_ou25_under": probabilities[:, 0], "p_ou25_over": probabilities[:, 1]}
    return {"p_kg_no": probabilities[:, 0], "p_kg_yes": probabilities[:, 1]}


def _log_loss_rows(y_true: pd.Series, probabilities: np.ndarray, class_order: tuple) -> np.ndarray:
    class_index = {label: index for index, label in enumerate(class_order)}
    encoded = np.asarray([class_index[value] for value in y_true.tolist()], dtype=int)
    chosen = probabilities[np.arange(len(encoded)), encoded]
    return -np.log(np.clip(chosen, 1e-15, 1.0))


def _identity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    if {"_source_dataset", "_source_row"}.issubset(frame.columns):
        out["match_id"] = (
            frame["_source_dataset"].astype("string")
            + ":"
            + frame["_source_row"].astype("string")
        )
        out["_source_dataset"] = frame["_source_dataset"].astype("string")
        out["_source_row"] = frame["_source_row"].astype("string")
    else:
        required = {"_date", "_team_home_canonical", "_team_away_canonical"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Cannot build stable match identity; missing columns: {missing}")
        out["match_id"] = (
            pd.to_datetime(frame["_date"]).dt.strftime("%Y-%m-%d")
            + ":"
            + frame["_team_home_canonical"].astype("string")
            + ":"
            + frame["_team_away_canonical"].astype("string")
        )
        out["_source_dataset"] = pd.NA
        out["_source_row"] = pd.NA

    for source, target in (
        ("_date", "_date"),
        ("_season", "_season"),
        ("_league_canonical", "_league"),
        ("_team_home_canonical", "home_team"),
        ("_team_away_canonical", "away_team"),
        ("home_score", "home_score"),
        ("away_score", "away_score"),
    ):
        out[target] = frame[source].to_numpy() if source in frame.columns else pd.NA
    return out.reset_index(drop=True)


def _partition_diagnostics(
    frame: pd.DataFrame,
    partition: str,
    models: dict[str, BaselineModel],
) -> tuple[pd.DataFrame, dict]:
    result = _identity_columns(frame)
    result.insert(0, "partition", partition)
    metrics = {"rows": int(len(frame)), "date_min": str(pd.to_datetime(frame["_date"]).min().date()),
               "date_max": str(pd.to_datetime(frame["_date"]).max().date()), "markets": {}}

    for market, target_column in TARGET_COLUMNS.items():
        model = models[market]
        probabilities = model.predict_proba(frame)
        class_order = tuple(model.classes_.tolist())
        y = frame[target_column]
        if y.isna().any():
            raise ValueError(f"{partition} contains missing {target_column} labels")
        for name, values in _probability_columns(market, probabilities).items():
            result[name] = values
        result[target_column] = y.to_numpy()
        row_losses = _log_loss_rows(y, probabilities, class_order)
        loss_column = f"ll_{market}"
        result[loss_column] = row_losses
        metrics["markets"][market] = {
            "metrics": evaluate_probabilities(y.to_numpy(), probabilities, class_order=class_order),
            "target_counts": {str(key): int(value) for key, value in y.value_counts().items()},
            "target_rates": {str(key): float(value) for key, value in y.value_counts(normalize=True).items()},
            "mean_row_log_loss": float(row_losses.mean()),
        }
        if market == "1x2":
            for label, column in zip(class_order, ("ll_1x2_if_H", "ll_1x2_if_D", "ll_1x2_if_A")):
                result[column] = np.where(y.to_numpy() == label, row_losses, np.nan)

    return result, metrics


def export_diagnostics(
    features_path: Path,
    search_report_path: Path,
    csv_path: Path,
    summary_path: Path,
    *,
    expected_test_rows: int | None = None,
) -> dict:
    frame = pd.read_csv(features_path, compression="infer", low_memory=False)
    required = {"_date", "_season", *FEATURES, *TARGET_COLUMNS.values()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature artifact is missing required columns: {missing}")

    search = json.loads(search_report_path.read_text(encoding="utf-8"))
    winners = {market: search["markets"][market]["validation_winner"] for market in TARGET_COLUMNS}
    if any(winner != "logistic_regression" for winner in winners.values()):
        raise ValueError(f"Fixed LR diagnostic export aborted; validation winners differ: {winners}")

    train, validation, test = chronological_split(frame)
    _, confirmation = _split_validation_by_date(validation)
    fit_test = pd.concat([train, validation], ignore_index=True).sort_values("_date", kind="stable")

    confirmation_models = {
        market: BaselineModel(market).fit(train, train[target])
        for market, target in TARGET_COLUMNS.items()
    }
    test_models = {
        market: BaselineModel(market).fit(fit_test, fit_test[target])
        for market, target in TARGET_COLUMNS.items()
    }

    confirmation_rows, confirmation_summary = _partition_diagnostics(
        confirmation, "validation_confirmation", confirmation_models
    )
    test_rows, test_summary = _partition_diagnostics(test, "final_test", test_models)

    if expected_test_rows is not None and len(test_rows) != expected_test_rows:
        raise ValueError(
            f"Final test row count changed: expected {expected_test_rows}, got {len(test_rows)}"
        )
    if test_rows["match_id"].duplicated().any():
        raise ValueError("Duplicate match_id values found in diagnostic export")

    combined = pd.concat([confirmation_rows, test_rows], ignore_index=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False, encoding="utf-8")

    summary = {
        "purpose": "Post-hoc row-level reconstruction for diagnosis only; no model or parameter selection.",
        "selected_models": winners,
        "feature_count": len(FEATURES),
        "fit_policy": {
            "confirmation": "LR refit on train only, matching validation confirmation evaluation.",
            "final_test": "LR refit on train+validation, matching the locked final holdout evaluation.",
        },
        "partitions": {
            "train": {"rows": int(len(train)), "date_min": str(pd.to_datetime(train["_date"]).min().date()),
                      "date_max": str(pd.to_datetime(train["_date"]).max().date())},
            "validation": {"rows": int(len(validation)), "date_min": str(pd.to_datetime(validation["_date"]).min().date()),
                           "date_max": str(pd.to_datetime(validation["_date"]).max().date())},
            "validation_confirmation": confirmation_summary,
            "final_test": test_summary,
        },
        "diagnostic_csv": csv_path.name,
        "diagnostic_rows": int(len(combined)),
        "note": "This export reconstructs predictions from the fixed pipeline to expose per-match probabilities. It is not a new model-selection run and must not be used to retune against the final test.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("search_report", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("audit-results/diagnostics_confirmation_test.csv"))
    parser.add_argument("--summary", type=Path, default=Path("audit-results/diagnostics_summary.json"))
    parser.add_argument("--expected-test-rows", type=int, default=1550)
    args = parser.parse_args()
    summary = export_diagnostics(
        args.features, args.search_report, args.csv, args.summary,
        expected_test_rows=args.expected_test_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
