"""EPL confirmation deep-dive into individual Elo feature contributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel, FEATURES
from zk2.models.lightgbm_search import _split_validation_by_date

ELO_FEATURES = (
    "elo_home_pre",
    "elo_away_pre",
    "elo_diff_pre",
    "elo_home_match_count",
    "elo_away_match_count",
)

# Leave-one-feature-out tests are accompanied by paired removals because
# elo_diff_pre is algebraically related to home/away ratings and can be
# partially substituted by the remaining columns.
ABLATIONS = {
    "full": (),
    "no_elo_home_pre": ("elo_home_pre",),
    "no_elo_away_pre": ("elo_away_pre",),
    "no_elo_diff_pre": ("elo_diff_pre",),
    "no_elo_home_and_away": ("elo_home_pre", "elo_away_pre"),
    "no_elo_home_match_count": ("elo_home_match_count",),
    "no_elo_away_match_count": ("elo_away_match_count",),
    "no_elo_match_counts": ("elo_home_match_count", "elo_away_match_count"),
    "no_all_elo": ELO_FEATURES,
}


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
    return [float(v) for v in np.quantile(estimates, [0.025, 0.975])]


def feature_sets() -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(feature for feature in FEATURES if feature not in removed)
        for name, removed in ABLATIONS.items()
    }


def run_elo_deep_dive(features_path: Path, csv_path: Path, json_path: Path, *, seed: int = 20261115) -> dict:
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
    if set(train["_y_1x2"].dropna().unique()) != {"H", "D", "A"}:
        raise ValueError("Training partition does not contain all 1X2 classes.")
    if epl["_y_1x2"].isna().any():
        raise ValueError("EPL confirmation contains missing 1X2 targets.")

    y = epl["_y_1x2"].to_numpy()
    full_model = BaselineModel("1x2", features=FEATURES).fit(train, train["_y_1x2"])
    full_p = full_model.predict_proba(epl)
    full_loss = _row_losses(y, full_p)
    full_pA = full_p[:, 2]

    comparison_rows = []
    probability_summary = {}
    predictions = {"full": full_p}
    losses = {"full": full_loss}

    for idx, (name, selected) in enumerate(feature_sets().items()):
        if name == "full":
            probs = full_p
            row_loss = full_loss
        else:
            model = BaselineModel("1x2", features=selected).fit(train, train["_y_1x2"])
            probs = model.predict_proba(epl)
            row_loss = _row_losses(y, probs)
        predictions[name] = probs
        losses[name] = row_loss
        delta_loss = row_loss - full_loss
        delta_pA = probs[:, 2] - full_pA
        ci = [0.0, 0.0] if name == "full" else _date_cluster_ci(
            epl["_date"], delta_loss, seed=seed + idx
        )
        is_away = y == "A"
        is_not_away = ~is_away
        comparison_rows.append({
            "ablation": name,
            "removed_features": ";".join(ABLATIONS[name]),
            "remaining_feature_count": len(selected),
            "log_loss": float(row_loss.mean()),
            "delta_log_loss_vs_full": float(delta_loss.mean()),
            "delta_ll_ci95_low": ci[0],
            "delta_ll_ci95_high": ci[1],
            "mean_pA_change_all": float(delta_pA.mean()),
            "mean_pA_change_actual_A": float(delta_pA[is_away].mean()) if is_away.any() else None,
            "mean_pA_change_actual_not_A": float(delta_pA[is_not_away].mean()) if is_not_away.any() else None,
            "mean_pA_full_actual_A": float(full_pA[is_away].mean()) if is_away.any() else None,
            "mean_pA_ablated_actual_A": float(probs[is_away, 2].mean()) if is_away.any() else None,
        })
        probability_summary[name] = {
            "mean_probability": {
                "H": float(probs[:, 0].mean()),
                "D": float(probs[:, 1].mean()),
                "A": float(probs[:, 2].mean()),
            },
            "mean_probability_when_actual_away_win": {
                "H": float(probs[is_away, 0].mean()) if is_away.any() else None,
                "D": float(probs[is_away, 1].mean()) if is_away.any() else None,
                "A": float(probs[is_away, 2].mean()) if is_away.any() else None,
            },
        }

    # Explicit feature redundancy diagnostics: report whether diff equals
    # home Elo minus away Elo up to floating-point tolerance in this sample.
    home = pd.to_numeric(epl["elo_home_pre"], errors="coerce").to_numpy()
    away = pd.to_numeric(epl["elo_away_pre"], errors="coerce").to_numpy()
    diff = pd.to_numeric(epl["elo_diff_pre"], errors="coerce").to_numpy()
    valid = np.isfinite(home) & np.isfinite(away) & np.isfinite(diff)
    identity_error = diff[valid] - (home[valid] - away[valid])

    result = {
        "purpose": "EPL-only validation-confirmation Elo feature ablation; final test is not loaded or evaluated.",
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
            "elo_feature_valid_rows": int(valid.sum()),
            "elo_diff_minus_home_minus_away_abs_max": float(np.max(np.abs(identity_error))) if valid.any() else None,
        },
        "elo_features": list(ELO_FEATURES),
        "comparisons": comparison_rows,
        "probability_summary": probability_summary,
        "bootstrap": {
            "method": "paired date-cluster bootstrap",
            "replicates": 2000,
            "seed": seed,
            "interpretation": "Positive delta log loss means removing the feature(s) worsened predictive log loss. CI entirely above zero supports measurable incremental contribution. Probability changes are ablated minus full-model probabilities.",
        },
        "method_note": (
            "Each model is fit on the same chronological training partition and evaluated on the same EPL confirmation rows. "
            "Single-feature leave-one-out results are conditional on the other Elo columns. Because elo_diff_pre may be derived "
            "from home and away Elo, paired removals are included to expose redundancy. These are predictive ablations, not causal effects."
        ),
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(csv_path, index=False, encoding="utf-8")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("audit-results/epl_elo_deep_dive.csv"))
    parser.add_argument("--json", type=Path, default=Path("audit-results/epl_elo_deep_dive.json"))
    args = parser.parse_args()
    result = run_elo_deep_dive(args.features, args.csv, args.json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
