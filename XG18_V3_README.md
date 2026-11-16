# XG18 v3 — Chronological ensemble challenger

V3 is isolated from the frozen baseline and V2. It does not change the production/default bot or open the sealed final test.

## Model design

- XGBoost + LightGBM + CatBoost, each tuned with Optuna on five expanding chronological folds inside the training partition.
- Ensemble weights are fit on training-only out-of-fold probabilities.
- Calibration candidates are none, temperature scaling, one-vs-rest Platt, and one-vs-rest isotonic. Candidate selection uses the last two chronological training OOF folds; the selected calibrator is then refit on training OOF predictions.
- Features extend V2's past-only rolling form with xG difference, goals-minus-xG, shot accuracy, goals per shot, points/xG momentum, venue-specific form, rest-day difference, Elo difference and H2H.
- All matches on a UTC date are featurized before results from that date update rolling state. Earlier validation-day results update state for later validation dates only; no validation outcome is used to fit models, weights, or calibration.

## Evaluation protocol

The existing 12,971 / 2,763 chronological split is preserved. V3 validation is a one-time challenger comparison against the frozen baseline and V2 on identical MatchIDs. The report includes overall accuracy, selective accuracy at P(max)>=0.65, coverage, Wilson intervals, log loss, Brier, and paired fixture bootstrap intervals for selective accuracy and coverage differences.

Targets of 80% selective accuracy and 15% coverage are goals, not guarantees. A positive point estimate with a bootstrap interval crossing zero is not evidence of superiority. Repeated tuning on this validation set would turn it into development data; do not do so. The final test remains sealed.

## Train

```bash
python -m pip install -r requirements-xg18-v3.txt
python scripts/train_xg18_v3.py --train data/xg18_train.csv --validation data/xg18_val.csv --outdir reports --trials 3
```

## Predict upcoming fixtures

```bash
python scripts/xg18_v3_bot.py --history data/match_history.csv --fixtures data/upcoming_fixtures.csv --model reports/xg18_v3.joblib --output reports/upcoming_predictions.csv --threshold 0.65
```

History must end before the single UTC fixture date supplied. The model is a CSV-driven predictor; it does not fetch live fixtures or team news.
