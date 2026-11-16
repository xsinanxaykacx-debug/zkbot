#!/usr/bin/env python3
"""Run the frozen XG18 1X2 ensemble on the untouched chronological validation set.

Fit policy:
- Team Poisson is fit on xg18_train.csv only.
- Elo predicts validation fixtures chronologically, batching all fixtures on each UTC date
  before applying any result from that date.
- Frozen ensemble weights are applied unchanged.
- No calibration, tuning, feature selection, or model selection uses validation outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SRC = ROOT / "frozen_src"
sys.path.insert(0, str(FROZEN_SRC))
import xg18_frozen_model as frozen  # noqa: E402

WEIGHTS = {"team_poisson": 0.13958888173927925, "elo": 0.8604111182607207}
K = 32.0
HOME_ADVANTAGE = 80.0
DAVIDSON_NU = 0.8
BANDS = [(0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.0000000001)]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / den
    return float(center - half), float(center + half)


def outcome_codes(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["FTHG"].to_numpy() > df["FTAG"].to_numpy(), 0,
                    np.where(df["FTHG"].to_numpy() == df["FTAG"].to_numpy(), 1, 2))


def summarize_band(conf: np.ndarray, correct: np.ndarray, lo: float, hi: float) -> dict:
    mask = (conf >= lo) & (conf < hi)
    n = int(mask.sum())
    wins = int(correct[mask].sum())
    low, high = wilson(wins, n)
    return {
        "band": f"{lo:.0%}-{hi:.0%}" if hi <= 1 else f"{lo:.0%}+",
        "n": n,
        "mean_confidence": float(conf[mask].mean()) if n else None,
        "correct": wins,
        "accuracy": wins / n if n else None,
        "wilson_95_low": low,
        "wilson_95_high": high,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    train_raw = pd.read_csv(args.train)
    val_raw = pd.read_csv(args.validation)
    required = {"League", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    for name, df in (("train", train_raw), ("validation", val_raw)):
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        if df[list(required)].isna().any().any():
            raise ValueError(f"{name} contains nulls in required identity/target columns")

    if len(train_raw) != 12971 or len(val_raw) != 2763:
        raise ValueError(f"Unexpected split sizes: train={len(train_raw)}, validation={len(val_raw)}")

    train = frozen.prep(train_raw)
    val = frozen.prep(val_raw)
    if train["DateParsed"].isna().any() or val["DateParsed"].isna().any():
        raise ValueError("Invalid dates found")
    if train["DateParsed"].max() >= val["DateParsed"].min():
        raise ValueError("Chronology violation: training is not strictly before validation")
    if set(train_raw["MatchID"].astype(str)) & set(val_raw["MatchID"].astype(str)):
        raise ValueError("Train/validation MatchID overlap")

    # Fit only on the training partition. Do not pass validation results to Team Poisson.
    poisson = frozen.TeamPoisson(alpha=10.0).fit(train)
    lam_home, lam_away = poisson.predict(val)
    poisson_joint = frozen.score_joint(lam_home, lam_away, rho=0.0, maxg=12)
    poisson_probs = frozen.metrics_joint(
        val["FTHG"].to_numpy(float), val["FTAG"].to_numpy(float), poisson_joint
    )["probs_1x2"]

    # Elo is a prequential replay: all fixtures on a UTC date are forecast before that
    # date's outcomes update ratings for subsequent dates.
    elo_probs = frozen.run_elo(
        train, val, K=K, ha=HOME_ADVANTAGE, nu=DAVIDSON_NU
    )
    if elo_probs.shape != (len(val), 3):
        raise ValueError(f"Unexpected Elo prediction shape: {elo_probs.shape}")

    probs = WEIGHTS["team_poisson"] * poisson_probs + WEIGHTS["elo"] * elo_probs
    probs = np.clip(probs, 1e-15, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    y = outcome_codes(val)
    pred = probs.argmax(axis=1)  # Frozen tie order: Home, Draw, Away.
    conf = probs.max(axis=1)
    correct = (pred == y).astype(int)

    labels = np.array(["H", "D", "A"])
    onehot = np.eye(3)[y]
    log_loss = float(-np.log(probs[np.arange(len(y)), y]).mean())
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    accuracy = float(correct.mean())
    ece = 0.0
    for i in range(10):
        m = (conf >= i / 10) & (conf < (i + 1) / 10 if i < 9 else conf <= 1.0)
        if m.any():
            ece += float(m.mean()) * abs(float(conf[m].mean()) - float(correct[m].mean()))

    bands = [summarize_band(conf, correct, lo, hi) for lo, hi in BANDS]
    selected = conf >= 0.60
    selected_n = int(selected.sum())
    selected_wins = int(correct[selected].sum())
    selected_low, selected_high = wilson(selected_wins, selected_n)

    out = pd.DataFrame({
        "MatchID": val["MatchID"].to_numpy() if "MatchID" in val else val_raw["MatchID"].to_numpy(),
        "Date": val_raw["Date"].to_numpy(),
        "League": val_raw["League"].to_numpy(),
        "HomeTeam": val_raw["HomeTeam"].to_numpy(),
        "AwayTeam": val_raw["AwayTeam"].to_numpy(),
        "P_H": probs[:, 0],
        "P_D": probs[:, 1],
        "P_A": probs[:, 2],
        "P_max": conf,
        "prediction": labels[pred],
        "actual": labels[y],
        "correct": correct,
        "lambda_home_team_poisson": lam_home,
        "lambda_away_team_poisson": lam_away,
    })
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.predictions, index=False)

    report = {
        "model": "XG18 frozen 1X2 TeamPoisson+Elo ensemble",
        "validation_rows": len(val),
        "training_rows": len(train),
        "validation_outcomes_used_for_fit_or_tuning": False,
        "elo_policy": "UTC-day batch forecast before same-day results update ratings",
        "weights": WEIGHTS,
        "elo_parameters": {"K": K, "home_advantage": HOME_ADVANTAGE, "davidson_nu": DAVIDSON_NU},
        "overall": {
            "accuracy": accuracy,
            "correct": int(correct.sum()),
            "n": len(correct),
            "wilson_95_low": wilson(int(correct.sum()), len(correct))[0],
            "wilson_95_high": wilson(int(correct.sum()), len(correct))[1],
            "log_loss": log_loss,
            "brier": brier,
            "top_label_ece_10bin": float(ece),
        },
        "pmax_ge_60": {
            "n": selected_n,
            "coverage": selected_n / len(val),
            "correct": selected_wins,
            "accuracy": selected_wins / selected_n if selected_n else None,
            "wilson_95_low": selected_low,
            "wilson_95_high": selected_high,
        },
        "confidence_bands": bands,
        "fit_diagnostics": {"team_poisson": poisson.fit_info},
        "prediction_file": str(args.predictions),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
