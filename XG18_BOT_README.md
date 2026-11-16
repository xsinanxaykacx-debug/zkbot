# XG18 1X2 Prediction Bot

This is an isolated implementation of the frozen XG18 Team Poisson + Elo 1X2 ensemble. It does not modify the frozen model source, its parameters, or weights.

## Validation result (one chronological validation run)

- Training: 12,971 matches.
- Validation: 2,763 matches (2025-10-05 through 2026-03-14).
- Validation outcomes were not used for fitting, calibration, tuning, or model selection.
- Overall 1X2 accuracy: **48.86%** (1,350 / 2,763), 95% Wilson CI 47.00%–50.72%.
- Predictions with model maximum probability >= 65%: **76.23% accuracy** (247 / 324), 95% Wilson CI 71.31%–80.54%, covering 11.73% of validation matches.
- Predictions with model maximum probability >= 60%: 69.08% accuracy (353 / 511), 95% Wilson CI 64.94%–72.93%, covering 18.49%.

Interpretation: the bot is a selective predictor, not a 65%-accurate predictor on every match. The measured 76.23% result applies only to the >=65% model-confidence subset on this historical validation set; it is not a guarantee for future fixtures. The sealed final test was not opened.

## Setup

Use Python 3.13 and install the pinned dependencies:

```bash
python -m pip install -r requirements-xg18-bot.txt
```

The bot assembles and verifies its frozen model source automatically from the committed archive parts on first run.

## Input files

History CSV must contain:
- `League`, `Season`, `Date`, `HomeTeam`, `AwayTeam`
- `FTHG`, `FTAG`

Fixtures CSV must contain:
- `League`, `Season`, `Date`, `HomeTeam`, `AwayTeam`

Use one upcoming UTC calendar date per invocation. The completed history must end before that UTC date. This restriction preserves the frozen Elo rule: all fixtures on a date are predicted before any results from that date are applied.

## Run

```bash
python scripts/xg18_bot.py \
  --history data/match_history.csv \
  --fixtures data/upcoming_fixtures.csv \
  --output reports/upcoming_predictions.csv \
  --confidence-threshold 0.65
```

The output contains Home/Draw/Away probabilities, the highest probability, the predicted outcome, and a boolean flag for rows whose maximum probability is at least the chosen threshold. The threshold is a probability filter, not a promise of hit rate.

## Reproduce the historical validation

The chronological evaluator is:

```bash
python scripts/evaluate_xg18_validation.py \
  --train data/xg18_train.csv \
  --validation data/xg18_val.csv \
  --predictions reports/xg18_validation_predictions.csv \
  --report reports/xg18_validation_report.json
```

It fits Team Poisson on training only and replays Elo chronologically with same-day batching. It applies the frozen ensemble weights unchanged and reports accuracy, log loss, Brier score, ECE, confidence bands, coverage, and Wilson 95% intervals.
