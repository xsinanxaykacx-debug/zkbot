"""Train source-aware ZK 2.0 market models without evaluating validation or final test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from zk2.data.splits import chronological_split
from zk2.features.pipeline import build_features
from zk2.models.baseline import MarketModelRouter, TARGET_COLUMNS


def train_market_models(dataset_path: Path, output_path: Path) -> dict:
    """Fit all market routers on the chronological training partition only."""
    raw = pd.read_csv(dataset_path, compression="infer", low_memory=False)
    required = {
        "_date", "_source_dataset", "_team_home_canonical",
        "_team_away_canonical", "_league_canonical", "MSE", "MSD",
        "xg_home", "xg_away", *TARGET_COLUMNS.values(),
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Enriched dataset is missing required columns: {missing}")

    featured = build_features(raw)
    train, _validation, _locked_final_test = chronological_split(featured)
    if train.empty:
        raise ValueError("Chronological training partition is empty.")

    models = {}
    for market in TARGET_COLUMNS:
        models[market] = MarketModelRouter(market).fit(train)

    metadata = {
        "model_family": "LogisticRegression",
        "training_rows": int(len(train)),
        "training_date_min": str(pd.to_datetime(train["_date"]).min().date()),
        "training_date_max": str(pd.to_datetime(train["_date"]).max().date()),
        "market_routing": {
            "1x2": "all sources; 32 baseline features; no xG",
            "ou25": "EPL.xlsm: 32 baseline + 16 rolling xG; non-EPL: 32 baseline",
            "kg": "EPL.xlsm: 32 baseline + 16 rolling xG; non-EPL: 32 baseline",
        },
        "xg_policy": (
            "Openplay match xG is used only for EPL.xlsm. Non-EPL models are "
            "trained separately without xG; no xG imputation is performed."
        ),
        "validation_policy": "Validation partition is not evaluated by this training script.",
        "final_test_policy": (
            "Locked final-test partition is not evaluated, scored, or used for model selection."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "metadata": metadata}, output_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/zk2_market_models_xg.joblib"),
    )
    args = parser.parse_args()
    print(json.dumps(train_market_models(args.dataset, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
