"""Controlled Elo-difference ablation and draw-probability diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel, FEATURES
from zk2.models.lightgbm_search import _split_validation_by_date

ELO_COLUMNS = (
    "elo_home_pre", "elo_away_pre", "elo_diff_pre",
    "elo_home_match_count", "elo_away_match_count",
)
ABLATIONS = {
    "full_elo_5": (),
    "without_elo_diff_4": ("elo_diff_pre",),
    "ratings_only_2": (
        "elo_diff_pre", "elo_home_match_count", "elo_away_match_count",
    ),
}
DRAW_BANDS = (-np.inf, 50.0, 100.0, 150.0, 200.0, np.inf)
DRAW_BAND_LABELS = ("0-50", "50-100", "100-150", "150-200", "200+")


def feature_sets() -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(feature for feature in FEATURES if feature not in removed)
        for name, removed in ABLATIONS.items()
    }


def _row_losses(y: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    class_index = {"H": 0, "D": 1, "A": 2}
    indices = np.asarray([class_index[value] for value in y], dtype=int)
    return -np.log(np.clip(probabilities[np.arange(len(indices)), indices], 1e-15, 1.0))


def _date_cluster_ci(
    dates: pd.Series, deltas: np.ndarray, *, seed: int, reps: int = 2000
) -> list[float]:
    date_values = pd.to_datetime(dates).dt.strftime("%Y-%m-%d").to_numpy()
    unique_dates = np.unique(date_values)
    if len(unique_dates) < 2:
        return [float("nan"), float("nan")]
    rows_by_date = {date: np.flatnonzero(date_values == date) for date in unique_dates}
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=float)
    for i in range(reps):
        sampled_dates = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        sampled_rows = np.concatenate([rows_by_date[date] for date in sampled_dates])
        estimates[i] = float(deltas[sampled_rows].mean())
    return [float(v) for v in np.quantile(estimates, [0.025, 0.975])]


def draw_band_summary(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> list[dict]:
    """Summarize P(D) by absolute pre-match Elo gap; no outcome leakage."""
    elo_gap = pd.to_numeric(frame["elo_diff_pre"], errors="coerce").abs()
    bands = pd.cut(
        elo_gap, bins=DRAW_BANDS, labels=DRAW_BAND_LABELS,
        right=False, include_lowest=True,
    )
    rows = []
    actual_draw = frame["_y_1x2"].to_numpy() == "D"
    for label in DRAW_BAND_LABELS:
        mask = (bands == label).to_numpy()
        if not mask.any():
            rows.append({
                "absolute_elo_gap_band": label, "matches": 0,
                "actual_draw_rate": None, "mean_pD_full": None,
                "mean_pD_without_diff": None, "mean_pD_ratings_only": None,
            })
            continue
        rows.append({
            "absolute_elo_gap_band": label,
            "matches": int(mask.sum()),
            "actual_draw_rate": float(actual_draw[mask].mean()),
            "mean_pD_full": float(predictions["full_elo_5"][mask, 1].mean()),
            "mean_pD_without_diff": float(predictions["without_elo_diff_4"][mask, 1].mean()),
            "mean_pD_ratings_only": float(predictions["ratings_only_2"][mask, 1].mean()),
        })
    return rows


def run_elo_diff_ablation(
    features_path: Path, csv_path: Path, json_path: Path, *, seed: int = 20261115
) -> dict:
    frame = pd.read_csv(features_path, compression="infer", low_memory=False)
    required = {"_date", "_source_dataset", "_y_1x2", *FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature artifact missing columns: {missing}")

    train, validation, _ = chronological_split(frame)
    _, confirmation = _split_validation_by_date(validation)
    source = confirmation["_source_dataset"].astype("string").str.lower()
    epl = confirmation.loc[source.str.contains("epl", na=False)].copy()
    if epl.empty:
        raise ValueError("No EPL.xlsm rows in validation confirmation partition.")
    if epl["_y_1x2"].isna().any():
        raise ValueError("EPL confirmation contains missing 1X2 targets.")
    if set(train["_y_1x2"].dropna().unique()) != {"H", "D", "A"}:
        raise ValueError("Training partition does not contain all 1X2 classes.")

    y = epl["_y_1x2"].to_numpy()
    predictions: dict[str, np.ndarray] = {}
    losses: dict[str, np.ndarray] = {}
    for name, selected in feature_sets().items():
        model = BaselineModel("1x2", features=selected).fit(train, train["_y_1x2"])
        probabilities = model.predict_proba(epl)
        predictions[name] = probabilities
        losses[name] = _row_losses(y, probabilities)

    rows = []
    class_rows = []
    full_loss = losses["full_elo_5"]
    for idx, name in enumerate(feature_sets()):
        delta = losses[name] - full_loss
        ci = [0.0, 0.0] if name == "full_elo_5" else _date_cluster_ci(
            epl["_date"], delta, seed=seed + idx
        )
        rows.append({
            "model": name,
            "removed_elo_columns": ";".join(ABLATIONS[name]),
            "feature_count": len(feature_sets()[name]),
            "log_loss": float(losses[name].mean()),
            "delta_ll_vs_full": float(delta.mean()),
            "delta_ll_ci95_low": ci[0],
            "delta_ll_ci95_high": ci[1],
        })
        for class_idx, label in enumerate(("H", "D", "A")):
            mask = y == label
            class_delta = delta[mask]
            class_ci = [0.0, 0.0] if name == "full_elo_5" else _date_cluster_ci(
                epl.loc[mask, "_date"], class_delta, seed=seed + 100 + idx * 10 + class_idx
            )
            class_rows.append({
                "model": name,
                "actual_class": label,
                "matches": int(mask.sum()),
                "conditional_log_loss": float(losses[name][mask].mean()),
                "delta_conditional_ll_vs_full": float(class_delta.mean()),
                "delta_ci95_low": class_ci[0],
                "delta_ci95_high": class_ci[1],
            })

    draw_rows = draw_band_summary(epl, predictions)
    home = pd.to_numeric(epl["elo_home_pre"], errors="coerce").to_numpy()
    away = pd.to_numeric(epl["elo_away_pre"], errors="coerce").to_numpy()
    diff = pd.to_numeric(epl["elo_diff_pre"], errors="coerce").to_numpy()
    valid = np.isfinite(home) & np.isfinite(away) & np.isfinite(diff)
    identity_error = diff[valid] - (home[valid] - away[valid])

    result = {
        "purpose": (
            "Controlled Elo feature ablation and draw-probability diagnostics on "
            "EPL validation-confirmation rows only; final test is not loaded."
        ),
        "training": {
            "rows": int(len(train)),
            "date_min": str(pd.to_datetime(train["_date"]).min().date()),
            "date_max": str(pd.to_datetime(train["_date"]).max().date()),
        },
        "evaluation": {
            "rows": int(len(epl)),
            "source": "EPL.xlsm",
            "date_min": str(pd.to_datetime(epl["_date"]).min().date()),
            "date_max": str(pd.to_datetime(epl["_date"]).max().date()),
            "class_counts": {label: int(np.sum(y == label)) for label in ("H", "D", "A")},
            "elo_identity_valid_rows": int(valid.sum()),
            "elo_diff_identity_abs_max": float(np.max(np.abs(identity_error))) if valid.any() else None,
        },
        "models": rows,
        "class_conditional_log_loss": class_rows,
        "draw_probability_by_absolute_elo_gap": draw_rows,
        "bootstrap": {
            "method": "paired date-cluster bootstrap",
            "replicates": 2000,
            "seed": seed,
            "interpretation": (
                "Positive delta means the ablated model has higher (worse) loss than full. "
                "Class-specific intervals resample dates containing that actual class."
            ),
        },
        "method_note": (
            "All variants use the same training rows, EPL confirmation rows, LR pipeline, "
            "train-fitted imputation/scaling and chronological split. ratings_only_2 retains "
            "elo_home_pre and elo_away_pre while removing elo_diff_pre and both Elo match counts. "
            "Draw bands use absolute pre-match Elo gap; actual draw rate is descriptive. "
            "Ablations are predictive comparisons, not causal effects."
        ),
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
    class_path = csv_path.with_name("elo_diff_class_log_loss.csv")
    pd.DataFrame(class_rows).to_csv(class_path, index=False, encoding="utf-8")
    draw_path = csv_path.with_name("elo_diff_draw_bands.csv")
    pd.DataFrame(draw_rows).to_csv(draw_path, index=False, encoding="utf-8")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("audit-results/elo_diff_ablation.csv"))
    parser.add_argument("--json", type=Path, default=Path("audit-results/elo_diff_ablation.json"))
    args = parser.parse_args()
    result = run_elo_diff_ablation(args.features, args.csv, args.json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
