#!/usr/bin/env python3
"""Self-contained CLI predictor for one upcoming UTC match date using frozen XG18 weights."""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SRC = ROOT / "frozen_src"
ARCHIVE_SHA256 = "d8d62f65b8887084a80d1da84ec6a758c401257172e50d5ef8a7137cc1dbc7aa"
MODULE_SHA256 = "79d89fa9491643590b96eb0e4a2761b0612a245247bd1fe40ed6b29710071e6e"
PARTS = [
    "research/xg18-frozen-source.zip.part00",
    "research/xg18-frozen-source.zip.part01",
    "research/xg18-frozen-source.zip.part02",
    "research/xg18-frozen-source.zip.part03",
    "research/xg18-frozen-source.zip.part04",
    "research/xg18-frozen-source.zip.part05",
    "research/xg18-frozen-source.zip.part06",
    "research/xg18-frozen-source.zip.part06b",
    "research/xg18-frozen-source.zip.part07",
    "research/xg18-frozen-source.zip.part08",
    "research/xg18-frozen-source.zip.part09",
]


def ensure_frozen_source() -> None:
    module = FROZEN_SRC / "xg18_frozen_model.py"
    if module.is_file() and hashlib.sha256(module.read_bytes()).hexdigest() == MODULE_SHA256:
        return
    archive = ROOT / "xg18-frozen-source.zip"
    archive.write_bytes(b"".join((ROOT / part).read_bytes() for part in PARTS))
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_hash != ARCHIVE_SHA256:
        raise RuntimeError(f"Frozen source archive hash mismatch: {archive_hash}")
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Frozen source ZIP CRC failure: {bad}")
        zf.extractall(FROZEN_SRC)
    actual = hashlib.sha256(module.read_bytes()).hexdigest()
    if actual != MODULE_SHA256:
        raise RuntimeError(f"Frozen model module hash mismatch: {actual}")


ensure_frozen_source()
sys.path.insert(0, str(FROZEN_SRC))
import xg18_frozen_model as frozen  # noqa: E402

W_TEAM_POISSON = 0.13958888173927925
W_ELO = 0.8604111182607207
OUTCOME_LABELS = np.array(["HOME", "DRAW", "AWAY"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict 1X2 probabilities for fixtures on one upcoming UTC date."
    )
    parser.add_argument("--history", type=Path, required=True,
                        help="Completed match history CSV with FTHG and FTAG.")
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="Upcoming fixtures CSV: League, Season, Date, HomeTeam, AwayTeam.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.65,
                        help="Flag predictions at or above this model probability (default 0.65).")
    args = parser.parse_args()

    if not 0.0 < args.confidence_threshold < 1.0:
        parser.error("--confidence-threshold must be between 0 and 1")

    history_raw = pd.read_csv(args.history)
    fixtures_raw = pd.read_csv(args.fixtures)
    required_history = {"League", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    required_fixtures = {"League", "Season", "Date", "HomeTeam", "AwayTeam"}
    missing_h = sorted(required_history - set(history_raw.columns))
    missing_f = sorted(required_fixtures - set(fixtures_raw.columns))
    if missing_h:
        parser.error(f"History CSV missing columns: {missing_h}")
    if missing_f:
        parser.error(f"Fixtures CSV missing columns: {missing_f}")
    if fixtures_raw.empty:
        parser.error("Fixtures CSV is empty")

    history = frozen.prep(history_raw)
    fixtures_input = fixtures_raw.copy()
    # Elo predicts all fixtures on the date before that date's results are applied.
    fixtures_input["FTHG"] = 0
    fixtures_input["FTAG"] = 0
    fixtures = frozen.prep(fixtures_input)

    if history["DateParsed"].isna().any() or fixtures["DateParsed"].isna().any():
        parser.error("Invalid dates found")
    fixture_days = fixtures["DateParsed"].dt.floor("D").unique()
    if len(fixture_days) != 1:
        parser.error("Supply fixtures from exactly one UTC calendar date per run.")
    if history["DateParsed"].dt.floor("D").max() >= fixture_days[0]:
        parser.error("History must end before the fixture UTC date; same-day results are not allowed.")
    if fixtures[["LeagueKey", "HomeKey", "AwayKey"]].duplicated().any():
        parser.error("Duplicate fixture rows detected")

    model = frozen.TeamPoisson(alpha=10.0).fit(history)
    lam_home, lam_away = model.predict(fixtures)
    joint = frozen.score_joint(lam_home, lam_away, rho=0.0, maxg=12)
    poisson_probs = frozen.metrics_joint(
        np.zeros(len(fixtures)), np.zeros(len(fixtures)), joint
    )["probs_1x2"]
    elo_probs = frozen.run_elo(history, fixtures, K=32.0, ha=80.0, nu=0.8)

    probs = W_TEAM_POISSON * poisson_probs + W_ELO * elo_probs
    probs = np.clip(probs, 1e-15, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    pmax = probs.max(axis=1)
    pred = probs.argmax(axis=1)

    result = pd.DataFrame({
        "Date": fixtures_raw["Date"].to_numpy(),
        "League": fixtures_raw["League"].to_numpy(),
        "HomeTeam": fixtures_raw["HomeTeam"].to_numpy(),
        "AwayTeam": fixtures_raw["AwayTeam"].to_numpy(),
        "P_HOME": probs[:, 0],
        "P_DRAW": probs[:, 1],
        "P_AWAY": probs[:, 2],
        "P_MAX": pmax,
        "PREDICTION": OUTCOME_LABELS[pred],
        "CONFIDENCE_AT_OR_ABOVE_THRESHOLD": pmax >= args.confidence_threshold,
        "THRESHOLD": args.confidence_threshold,
        "LAMBDA_HOME": lam_home,
        "LAMBDA_AWAY": lam_away,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Rows: {len(result)} | flagged >= {args.confidence_threshold:.0%}: "
          f"{int((pmax >= args.confidence_threshold).sum())}")
    print("The confidence threshold is not a guaranteed hit rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
