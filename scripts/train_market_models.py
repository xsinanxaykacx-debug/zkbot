"""Train source-aware ZK 2.0 market models without evaluating validation or final test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import pandas as pd

from zk2.data.splits import chronological_split
from zk2.features.pipeline import build_features
from zk2.models.baseline import MarketModelRouter, TARGET_COLUMNS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_metadata(frame: pd.DataFrame) -> dict:
    dates = pd.to_datetime(frame["_date"], errors="coerce").dropna()
    return {
        "rows": int(len(frame)),
        "date_min": str(dates.min().date()) if not dates.empty else None,
        "date_max": str(dates.max().date()) if not dates.empty else None,
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def train_market_models(
    dataset_path: Path,
    output_path: Path,
    *,
    locked_dataset_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict:
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
    train, validation, locked_final_test = chronological_split(featured)
    if train.empty:
        raise ValueError("Chronological training partition is empty.")

    models = {}
    for market in TARGET_COLUMNS:
        models[market] = MarketModelRouter(market).fit(train)

    locked_dataset_sha256 = _sha256(locked_dataset_path) if locked_dataset_path else None
    manifest_sha256 = _sha256(manifest_path) if manifest_path else None
    manifest_dataset_sha256 = None
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_dataset_sha256 = manifest.get("combined", {}).get("sha256")
        if locked_dataset_sha256 and manifest_dataset_sha256 != locked_dataset_sha256:
            raise ValueError("Locked dataset SHA-256 does not match the supplied manifest.")

    metadata = {
        "model_family": "LogisticRegression",
        "runtime": {
            "python_version": sys.version,
            "packages": {
                name: _package_version(name)
                for name in ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "lightgbm")
            },
        },
        "dataset_integrity": {
            "locked_dataset_path": str(locked_dataset_path) if locked_dataset_path else None,
            "locked_dataset_sha256": locked_dataset_sha256,
            "xg_enriched_dataset_path": str(dataset_path),
            "xg_enriched_dataset_sha256": _sha256(dataset_path),
            "manifest_path": str(manifest_path) if manifest_path else None,
            "manifest_sha256": manifest_sha256,
            "manifest_locked_dataset_sha256": manifest_dataset_sha256,
        },
        "split": {
            "method": "chronological_split by unique match dates",
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "final_test_fraction": 0.15,
            "train": _partition_metadata(train),
            "validation": _partition_metadata(validation),
            "final_test": _partition_metadata(locked_final_test),
        },
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
    parser.add_argument("--locked-dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        train_market_models(
            args.dataset, args.output,
            locked_dataset_path=args.locked_dataset,
            manifest_path=args.manifest,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
