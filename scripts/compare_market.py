"""Compare ZK 2.0 with de-vigged bookmaker odds on the same EPL test matches.

The current DATA.csv odds do not identify a verified bookmaker/time snapshot.
Therefore this comparison is deliberately restricted to EPL rows, where the
adapter maps bookmaker-specific opening and closing odds explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zk2.data.splits import chronological_split
from zk2.models.baseline import BaselineModel, TARGET_COLUMNS
from zk2.models.market import implied_probabilities
from zk2.models.metrics import evaluate_probabilities

BOOKMAKERS = {
    "pinnacle_open": {
        "1x2": ("_odd_H_pinnacle_open", "_odd_D_pinnacle_open", "_odd_A_pinnacle_open"),
        "ou25": ("_odd_under25_pinnacle_open", "_odd_over25_pinnacle_open"),
    },
    "pinnacle_close": {
        "1x2": ("_odd_H_pinnacle_close", "_odd_D_pinnacle_close", "_odd_A_pinnacle_close"),
        "ou25": ("_odd_under25_pinnacle_close", "_odd_over25_pinnacle_close"),
    },
    "bet365_open": {
        "1x2": ("_odd_H_bet365_open", "_odd_D_bet365_open", "_odd_A_bet365_open"),
        "ou25": ("_odd_under25_bet365_open", "_odd_over25_bet365_open"),
    },
    "bet365_close": {
        "1x2": ("_odd_H_bet365_close", "_odd_D_bet365_close", "_odd_A_bet365_close"),
        "ou25": ("_odd_under25_bet365_close", "_odd_over25_bet365_close"),
    },
}


def compare_market(features_path: Path, report_path: Path) -> dict:
    frame = pd.read_csv(features_path, compression="infer", low_memory=False)
    train, _, test = chronological_split(frame)
    epl_test = test.loc[test["_source_dataset"].eq("EPL.xlsm")].copy()
    if epl_test.empty:
        raise ValueError("No EPL matches in chronological test partition.")

    report: dict = {
        "scope": "EPL-only chronological test cohort",
        "test_period": {
            "date_min": str(pd.to_datetime(epl_test["_date"]).min().date()),
            "date_max": str(pd.to_datetime(epl_test["_date"]).max().date()),
        },
        "epl_test_rows": int(len(epl_test)),
        "method": "Proportional de-vig: (1/odds) / sum(1/odds); model and bookmaker scored on identical rows per source/market.",
        "markets": {},
        "limitations": [
            "DATA.csv odds are excluded because the source/bookmaker and timestamp are not verified in the current schema.",
            "No bookmaker KG odds are mapped; KG market comparison is unavailable.",
            "This is a log-loss/Brier comparison, not a profitability or CLV test.",
        ],
    }

    for market, target_column in TARGET_COLUMNS.items():
        model = BaselineModel(market)
        model.fit(train, train[target_column])
        model_probabilities = model.predict_proba(epl_test)
        class_order = tuple(model.classes_.tolist())
        valid_target = epl_test[target_column].notna().to_numpy()
        market_results = {
            "model": None,
            "bookmakers": {},
        }

        if valid_target.any():
            market_results["model"] = evaluate_probabilities(
                epl_test.loc[valid_target, target_column].to_numpy(),
                model_probabilities[valid_target],
                class_order=class_order,
            )

        if market == "kg":
            market_results["unavailable_reason"] = "No mapped bookmaker KG odds in EPL.xlsm."
            report["markets"][market] = market_results
            continue

        for bookmaker, market_columns in BOOKMAKERS.items():
            columns = market_columns.get(market)
            if columns is None or not all(column in epl_test.columns for column in columns):
                continue
            odds = epl_test.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            valid_odds = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
            paired = valid_target & valid_odds
            if not paired.any():
                market_results["bookmakers"][bookmaker] = {"n": 0, "reason": "No complete valid odds rows."}
                continue
            market_probs = implied_probabilities(odds[paired])
            y = epl_test.loc[paired, target_column].to_numpy()
            model_same_rows = model_probabilities[paired]
            market_metrics = evaluate_probabilities(y, market_probs, class_order=class_order)
            model_metrics = evaluate_probabilities(y, model_same_rows, class_order=class_order)
            market_results["bookmakers"][bookmaker] = {
                "n": int(paired.sum()),
                "model_same_rows": model_metrics,
                "bookmaker": market_metrics,
                "log_loss_delta_model_minus_market": round(
                    model_metrics["log_loss"] - market_metrics["log_loss"], 8
                ),
                "brier_delta_model_minus_market": round(
                    model_metrics["brier"] - market_metrics["brier"], 8
                ),
            }
        report["markets"][market] = market_results

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--report", type=Path, default=Path("zk2_market_comparison.json"))
    args = parser.parse_args()
    print(json.dumps(compare_market(args.features, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
