"""EPL-only confirmation ablation for the fixed 1X2 logistic baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel, FEATURES
from zk2.models.lightgbm_search import _split_validation_by_date

GROUPS = {
    "elo": tuple(f for f in FEATURES if f.startswith("elo_")),
    "form": tuple(f for f in FEATURES if f.startswith("form_")),
    "h2h": tuple(f for f in FEATURES if f.startswith("h2h_")),
    "league": tuple(f for f in FEATURES if f.startswith("league_")),
}


def feature_sets() -> dict[str, tuple[str, ...]]:
    sets = {"full": tuple(FEATURES)}
    for group, columns in GROUPS.items():
        sets[f"no_{group}"] = tuple(f for f in FEATURES if f not in columns)
    return sets


def _row_losses(y: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    class_index = {"H": 0, "D": 1, "A": 2}
    indices = np.asarray([class_index[value] for value in y], dtype=int)
    return -np.log(np.clip(probabilities[np.arange(len(indices)), indices], 1e-15, 1.0))


def _date_cluster_ci(dates: pd.Series, deltas: np.ndarray, *, seed: int, reps: int = 2000) -> list[float]:
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
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def run_ablation(features_path: Path, report_path: Path, output_path: Path, *, seed: int = 20261115) -> dict:
    frame = pd.read_csv(features_path, compression="infer", low_memory=False)
    required = {"_date", "_source_dataset", "_y_1x2", *FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature artifact missing columns: {missing}")

    train, validation, _ = chronological_split(frame)
    _, confirmation = _split_validation_by_date(validation)
    source = confirmation["_source_dataset"].astype("string").str.lower()
    epl_mask = source.str.contains("epl", na=False)
    epl_confirmation = confirmation.loc[epl_mask].copy()
    if epl_confirmation.empty:
        raise ValueError("No EPL.xlsm rows found in validation confirmation partition.")
    if set(train["_y_1x2"].dropna().unique()) != {"H", "D", "A"}:
        raise ValueError("Training partition does not contain all 1X2 classes.")

    sets = feature_sets()
    predictions: dict[str, np.ndarray] = {}
    losses: dict[str, np.ndarray] = {}
    metrics = {}
    y = epl_confirmation["_y_1x2"].to_numpy()
    for name, selected_features in sets.items():
        model = BaselineModel("1x2", features=selected_features)
        model.fit(train, train["_y_1x2"])
        probabilities = model.predict_proba(epl_confirmation)
        predictions[name] = probabilities
        losses[name] = _row_losses(y, probabilities)
        metrics[name] = {
            "feature_count": len(selected_features),
            "log_loss": float(log_loss(y, probabilities, labels=["H", "D", "A"])),
            "class_counts": {label: int(np.sum(y == label)) for label in ("H", "D", "A")},
            "class_log_loss": {
                label: float(losses[name][y == label].mean()) for label in ("H", "D", "A")
            },
        }

    rows = []
    full_losses = losses["full"]
    for index, (name, selected_features) in enumerate(sets.items()):
        delta = losses[name] - full_losses
        ci = [0.0, 0.0] if name == "full" else _date_cluster_ci(
            epl_confirmation["_date"], delta, seed=seed + index
        )
        rows.append({
            "feature_set": name,
            "feature_count": len(selected_features),
            "log_loss": metrics[name]["log_loss"],
            "delta_vs_full": float(delta.mean()),
            "delta_ci95_low": ci[0],
            "delta_ci95_high": ci[1],
        })

    result = {
        "purpose": "EPL-only validation-confirmation feature ablation; final test is not loaded or evaluated.",
        "training": {
            "rows": int(len(train)),
            "source_mix": {str(k): int(v) for k, v in train["_source_dataset"].value_counts().items()},
            "date_min": str(pd.to_datetime(train["_date"]).min().date()),
            "date_max": str(pd.to_datetime(train["_date"]).max().date()),
        },
        "evaluation": {
            "rows": int(len(epl_confirmation)),
            "source": "EPL.xlsm",
            "date_min": str(pd.to_datetime(epl_confirmation["_date"]).min().date()),
            "date_max": str(pd.to_datetime(epl_confirmation["_date"]).max().date()),
            "class_counts": {label: int(np.sum(y == label)) for label in ("H", "D", "A")},
        },
        "feature_groups": {key: list(value) for key, value in GROUPS.items()},
        "results": metrics,
        "paired_date_cluster_bootstrap": {
            "replicates": 2000,
            "seed": seed,
            "interpretation": "Positive delta means removing the feature group worsened LL; CI entirely above zero supports a measurable contribution.",
            "comparisons": rows,
        },
        "caution": "This is a validation-confirmation ablation, not a new final-test evaluation. Feature-group ablations are correlated and do not establish causal effects.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("audit-results/epl_feature_ablation.csv"))
    parser.add_argument("--json", type=Path, default=Path("audit-results/epl_feature_ablation.json"))
    args = parser.parse_args()
    result = run_ablation(args.features, args.json, args.csv)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
