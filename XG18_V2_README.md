# XG18 v2 — Pre-match Form Model

This is an isolated challenger to the frozen XG18 Team Poisson + Elo baseline. It does not replace or modify the baseline bot.

## Design

- Generates 106 pre-match features from prior completed matches: rolling last-3/5/10 team form, venue-specific form, goals, xG, shots, shots on target, big chances, points, rest days, Elo, head-to-head history, and prior league scoring averages.
- All fixtures on the same UTC date are featurized before any result from that date updates team state, Elo, or head-to-head state.
- Current-match shots, xG, goals, cards, possession, corners, attendance, and other post-match values are never used as features for that match.
- Injury, referee, squad-value, and weather fields are excluded because this dataset does not establish when those values became available before kickoff.
- Optuna tunes XGBoost using chronological folds inside the training partition only.
- Temperature calibration is fitted from training-only out-of-fold predictions.
- Validation outcomes are not used for feature generation, tuning, or calibration.

## Current development comparison

Dataset: 12,971 training matches and 2,763 chronological validation matches. The sealed final test remains unopened.

| Metric | Frozen baseline | v2 candidate |
|---|---:|---:|
| Overall accuracy | 48.86% | 49.84% |
| P(max) >= 65% | 247 / 324 = 76.23% | 245 / 318 = 77.04% |
| Coverage | 11.73% | 11.51% |
| Log loss | 1.01400 | 1.01017 |
| Brier | 0.60746 | 0.60466 |

The v2 selective-accuracy point estimate is +0.81 percentage points, but a paired bootstrap gives a 95% interval of -2.72 to +4.40 points. This does **not** establish statistical superiority. Coverage is also slightly lower. Therefore v2 remains an experimental challenger; the frozen baseline remains the reference and is not replaced.

## Train

```bash
python -m pip install -r requirements-xg18-v2.txt
python scripts/train_xg18_v2.py --train data/xg18_train.csv --validation data/xg18_val.csv --outdir reports --trials 4
```

The script writes `xg18_v2.joblib`, `xg18_v2_model_metadata.json`, validation predictions, and a report.

## Predict

History CSV must contain `League, Season, Date, HomeTeam, AwayTeam, FTHG, FTAG` plus historical match-stat columns used to build rolling form. Fixtures CSV must contain `League, Season, Date, HomeTeam, AwayTeam`. Supply exactly one upcoming UTC date per run; history must end before that date.

```bash
python scripts/xg18_v2_bot.py --history data/match_history.csv --fixtures data/upcoming_fixtures.csv --model reports/xg18_v2.joblib --output reports/upcoming_predictions.csv --threshold 0.65
```

## Reproducibility and safety

- Fixed random seeds are used for Optuna and XGBoost.
- The same-day leakage test is in `tests/test_prematch_features.py`.
- `scripts/compare_xg18_v2_baseline.py` compares baseline and challenger on identical MatchIDs and bootstraps selective-accuracy and coverage differences.
- Do not tune repeatedly against the same validation set and then describe it as an untouched test. The final test remains sealed.
