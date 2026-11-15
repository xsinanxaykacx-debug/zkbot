"""Reload the trained ZK 2.0 model bundle in a fresh process and smoke-test predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from zk2.models.baseline import (
    CLASS_ORDER,
    EPL_SOURCE_DATASET,
    FEATURES,
    XG_FEATURES,
    MarketModelRouter,
    TARGET_COLUMNS,
)


def smoke_test(model_path: Path) -> None:
    # This script is invoked as a separate Python process after training.
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise TypeError("Model artifact must be a dictionary.")
    if set(("models", "metadata")).difference(artifact):
        raise AssertionError("Model artifact is missing models or metadata.")

    models = artifact["models"]
    expected_markets = set(TARGET_COLUMNS)
    if set(models) != expected_markets:
        raise AssertionError(
            f"Expected markets {sorted(expected_markets)}, got {sorted(models)}."
        )

    # Include both source routes so OU/KG exercise EPL+xG and non-EPL paths.
    rows = pd.DataFrame(
        np.zeros((2, len(FEATURES) + len(XG_FEATURES)), dtype=float),
        columns=list(FEATURES + XG_FEATURES),
    )
    rows["_source_dataset"] = [EPL_SOURCE_DATASET, "data.csv"]

    for market in ("1x2", "ou25", "kg"):
        model = models[market]
        if not isinstance(model, MarketModelRouter):
            raise TypeError(f"{market} is not a MarketModelRouter.")
        if not model._is_fitted:
            raise AssertionError(f"{market} router is not marked fitted.")

        probabilities = np.asarray(model.predict_proba(rows), dtype=float)
        expected_classes = len(CLASS_ORDER[market])
        expected_shape = (2, expected_classes)
        if probabilities.shape != expected_shape:
            raise AssertionError(
                f"{market}: expected shape {expected_shape}, got {probabilities.shape}."
            )
        if not np.isfinite(probabilities).all():
            raise AssertionError(f"{market}: predictions contain non-finite values.")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise AssertionError(f"{market}: probabilities are outside [0, 1].")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise AssertionError(f"{market}: probability rows do not sum to 1.")

        print(
            f"{market}: OK — shape={probabilities.shape}; "
            "EPL and non-EPL routes exercised"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=Path("artifacts/zk2_market_models_xg.joblib"),
    )
    args = parser.parse_args()
    smoke_test(args.model)


if __name__ == "__main__":
    main()
