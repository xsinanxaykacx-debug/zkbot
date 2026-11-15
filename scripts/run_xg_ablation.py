"""Compare fixed LR baselines with/without rolling xG on EPL confirmation only.

The locked final test partition is deliberately not evaluated by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zk2.data.splits import chronological_split
from zk2.data.epl_team_mapping import normalize_epl_team
from zk2.data.team_mapping import team_key
from zk2.features.pipeline import build_features
from zk2.models.baseline import BaselineModel, FEATURES, TARGET_COLUMNS
from zk2.models.lightgbm_search import _split_validation_by_date


def _match_key(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["_date"], errors="coerce").dt.normalize()
    home = frame["_team_home_canonical"].map(
        lambda value: team_key(normalize_epl_team(value))
    )
    away = frame["_team_away_canonical"].map(
        lambda value: team_key(normalize_epl_team(value))
    )
    return dates.dt.strftime("%Y-%m-%d") + "|" + home + "|" + away


def _xg_feature_columns() -> tuple[str, ...]:
    columns = []
    for side in ("home", "away"):
        for window in (5, 10):
            for metric in ("for", "against", "diff"):
                columns.append(f"xg_form_{side}_{metric}_avg_{window}")
            columns.append(f"xg_form_{side}_matches_{window}")
    return tuple(columns)


def _row_scores(
    y: np.ndarray, probabilities: np.ndarray, class_order: tuple
) -> tuple[np.ndarray, np.ndarray]:
    class_index = {label: index for index, label in enumerate(class_order)}
    encoded = np.asarray([class_index[value] for value in y], dtype=int)
    chosen = probabilities[np.arange(len(encoded)), encoded]
    log_loss_rows = -np.log(np.clip(chosen, 1e-15, 1.0))
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(encoded)), encoded] = 1.0
    if len(class_order) == 2:
        brier_rows = (probabilities[:, 1] - one_hot[:, 1]) ** 2
    else:
        brier_rows = np.sum((probabilities - one_hot) ** 2, axis=1)
    return log_loss_rows, brier_rows


def _date_cluster_ci(
    dates: pd.Series,
    deltas: np.ndarray,
    *,
    seed: int,
    reps: int = 2000,
) -> list[float]:
    date_values = pd.to_datetime(dates).dt.strftime("%Y-%m-%d").to_numpy()
    unique_dates = np.unique(date_values)
    if len(unique_dates) < 2:
        raise ValueError("At least two distinct evaluation dates are required.")
    rows_by_date = {
        date: np.flatnonzero(date_values == date) for date in unique_dates
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=float)
    for index in range(reps):
        sampled_dates = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        sampled_rows = np.concatenate([rows_by_date[date] for date in sampled_dates])
        estimates[index] = float(deltas[sampled_rows].mean())
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def run_xg_ablation(
    dataset_path: Path,
    openplay_audit_path: Path,
    csv_path: Path,
    report_path: Path,
    *,
    enriched_dataset_path: Path | None = None,
    seed: int = 20261115,
    bootstrap_reps: int = 2000,
) -> dict:
    raw = pd.read_csv(dataset_path, compression="infer", low_memory=False)
    audit = pd.read_csv(openplay_audit_path, low_memory=False)
    required_raw = {
        "_date", "_source_dataset", "_team_home_canonical",
        "_team_away_canonical", "MSE", "MSD", *TARGET_COLUMNS.values(),
    }
    missing_raw = sorted(required_raw.difference(raw.columns))
    if missing_raw:
        raise ValueError(f"Locked dataset is missing columns: {missing_raw}")
    required_audit = {
        "match_key", "xg_home", "xg_away", "openplay_matched",
        "score_match", "clean_xg_match",
    }
    missing_audit = sorted(required_audit.difference(audit.columns))
    if missing_audit:
        raise ValueError(f"Openplay audit is missing columns: {missing_audit}")
    if audit["match_key"].duplicated().any():
        raise ValueError("Openplay audit contains duplicate match keys.")

    epl_mask = raw["_source_dataset"].astype("string").str.casefold().eq("epl.xlsm")
    epl = raw.loc[epl_mask].copy()
    epl["_xg_row_index"] = epl.index
    epl["match_key"] = _match_key(epl)
    if epl["match_key"].duplicated().any():
        raise ValueError("Locked EPL source contains duplicate match keys.")

    audit_columns = [
        "match_key", "xg_home", "xg_away", "openplay_matched",
        "score_match", "clean_xg_match",
    ]
    joined = epl[["_xg_row_index", "match_key"]].merge(
        audit[audit_columns], on="match_key", how="left",
        validate="one_to_one", indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Openplay audit keys do not cover every EPL source row.")
    matched = joined["openplay_matched"].fillna(False).astype(bool)
    score_ok = joined["score_match"].fillna(False).astype(bool)
    clean = joined["clean_xg_match"].fillna(False).astype(bool)
    if int((matched & ~score_ok).sum()) != 0:
        raise ValueError("Openplay score mismatch detected; xG experiment aborted.")
    if int((clean & (~matched | ~score_ok)).sum()) != 0:
        raise ValueError("Clean xG flags are inconsistent with match/score checks.")

    joined["xg_home"] = pd.to_numeric(joined["xg_home"], errors="coerce")
    joined["xg_away"] = pd.to_numeric(joined["xg_away"], errors="coerce")
    if clean.any() and joined.loc[clean, ["xg_home", "xg_away"]].isna().any().any():
        raise ValueError("A clean xG match has missing home/away xG.")

    raw["xg_home"] = np.nan
    raw["xg_away"] = np.nan
    row_indices = joined["_xg_row_index"].to_numpy()
    raw.loc[row_indices, "xg_home"] = joined["xg_home"].to_numpy()
    raw.loc[row_indices, "xg_away"] = joined["xg_away"].to_numpy()

    # Keep the v1 locked dataset immutable; publish a separate xG-enriched
    # derivative for the league-aware production-model training path.
    if enriched_dataset_path is not None:
        enriched_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(
            enriched_dataset_path,
            index=False,
            encoding="utf-8",
            compression={"method": "gzip", "mtime": 0},
        )

    featured = build_features(raw)
    train, validation, _locked_final_test = chronological_split(featured)
    _, confirmation = _split_validation_by_date(validation)
    train_epl = train.loc[
        train["_source_dataset"].astype("string").str.casefold().eq("epl.xlsm")
    ].copy()
    confirmation_epl = confirmation.loc[
        confirmation["_source_dataset"].astype("string").str.casefold().eq("epl.xlsm")
    ].copy()
    if train_epl.empty or confirmation_epl.empty:
        raise ValueError("EPL-only training or confirmation partition is empty.")

    xg_features = _xg_feature_columns()
    missing_xg_features = sorted(set(xg_features).difference(featured.columns))
    if missing_xg_features:
        raise ValueError(f"Rolling xG feature generation failed: {missing_xg_features}")

    results = {}
    paired_rows = []
    for market_index, (market, target_column) in enumerate(TARGET_COLUMNS.items()):
        train_market = train_epl.loc[train_epl[target_column].notna()].copy()
        eval_market = confirmation_epl.loc[confirmation_epl[target_column].notna()].copy()
        if eval_market.empty:
            raise ValueError(f"No valid {market} confirmation targets.")
        baseline_model = BaselineModel(market, features=tuple(FEATURES))
        xg_model = BaselineModel(market, features=tuple(FEATURES) + xg_features)
        baseline_model.fit(train_market, train_market[target_column])
        xg_model.fit(train_market, train_market[target_column])

        y = eval_market[target_column].to_numpy()
        class_order = tuple(baseline_model.classes_.tolist())
        baseline_prob = baseline_model.predict_proba(eval_market)
        xg_prob = xg_model.predict_proba(eval_market)
        baseline_ll, baseline_brier = _row_scores(y, baseline_prob, class_order)
        xg_ll, xg_brier = _row_scores(y, xg_prob, class_order)
        ll_delta = xg_ll - baseline_ll
        brier_delta = xg_brier - baseline_brier
        ll_ci = _date_cluster_ci(
            eval_market["_date"], ll_delta,
            seed=seed + market_index * 10, reps=bootstrap_reps,
        )
        brier_ci = _date_cluster_ci(
            eval_market["_date"], brier_delta,
            seed=seed + market_index * 10 + 1, reps=bootstrap_reps,
        )
        results[market] = {
            "train_rows": int(len(train_market)),
            "evaluation_rows": int(len(eval_market)),
            "class_order": list(class_order),
            "baseline": {
                "log_loss": float(baseline_ll.mean()),
                "brier": float(baseline_brier.mean()),
            },
            "with_xg": {
                "feature_count": len(FEATURES) + len(xg_features),
                "log_loss": float(xg_ll.mean()),
                "brier": float(xg_brier.mean()),
            },
            "paired_delta_xg_minus_baseline": {
                "log_loss": {
                    "mean": float(ll_delta.mean()),
                    "ci95": ll_ci,
                    "interpretation": "Negative favours rolling xG; CI entirely below zero supports improvement.",
                },
                "brier": {
                    "mean": float(brier_delta.mean()),
                    "ci95": brier_ci,
                    "interpretation": "Negative favours rolling xG; CI entirely below zero supports improvement.",
                },
            },
        }
        for position, (_, row) in enumerate(eval_market.iterrows()):
            paired_rows.append({
                "market": market,
                "date": row["_date"],
                "home_team": row["_team_home_canonical"],
                "away_team": row["_team_away_canonical"],
                "target": y[position],
                "baseline_log_loss": baseline_ll[position],
                "xg_log_loss": xg_ll[position],
                "delta_log_loss_xg_minus_baseline": ll_delta[position],
                "baseline_brier": baseline_brier[position],
                "xg_brier": xg_brier[position],
                "delta_brier_xg_minus_baseline": brier_delta[position],
            })

    report = {
        "purpose": "EPL-only LR comparison with/without rolling xG; validation confirmation only.",
        "source": {
            "xg_provider": "Openplay data-v0.1.0",
            "attribution": "openplay (Noah Bair), CC BY 4.0",
            "epl_source_rows": int(len(epl)),
            "clean_xg_matches": int(clean.sum()),
            "unmatched_or_unverified_rows": int((~clean).sum()),
            "score_mismatches": int((matched & ~score_ok).sum()),
            "missing_xg_policy": "Null remains null; no zero filling. Rolling windows count prior xG-observed matches only.",
        },
        "split": {
            "train_rows_all_sources": int(len(train)),
            "train_epl_rows": int(len(train_epl)),
            "confirmation_rows_all_sources": int(len(confirmation)),
            "confirmation_epl_rows": int(len(confirmation_epl)),
            "train_date_min": str(pd.to_datetime(train_epl["_date"]).min().date()),
            "train_date_max": str(pd.to_datetime(train_epl["_date"]).max().date()),
            "confirmation_date_min": str(pd.to_datetime(confirmation_epl["_date"]).min().date()),
            "confirmation_date_max": str(pd.to_datetime(confirmation_epl["_date"]).max().date()),
            "final_test_policy": "Final test partition is not evaluated, scored, or used for model selection.",
        },
        "feature_policy": {
            "baseline_feature_count": len(FEATURES),
            "rolling_xg_feature_count": len(xg_features),
            "rolling_xg_features": list(xg_features),
            "windows": [5, 10],
            "timing": "Pre-match; all same-calendar-day fixtures use day-start snapshots; results update after the day.",
        },
        "bootstrap": {
            "method": "Paired calendar-date cluster bootstrap",
            "replicates": bootstrap_reps,
            "seed": seed,
            "same_rows_for_both_models": True,
        },
        "markets": results,
        "caution": "This is a validation-confirmation experiment only. Do not tune against the locked final test; a positive result does not establish profitability or guarantee a 0.65-0.67 log loss.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(paired_rows).to_csv(csv_path, index=False, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("openplay_audit", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("audit-results/xg_lr_paired_rows.csv"))
    parser.add_argument("--report", type=Path, default=Path("audit-results/xg_lr_ablation.json"))
    parser.add_argument("--enriched-dataset", type=Path)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args()
    report = run_xg_ablation(
        args.dataset, args.openplay_audit, args.csv, args.report,
        enriched_dataset_path=args.enriched_dataset,
        seed=args.seed, bootstrap_reps=args.bootstrap_reps,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
