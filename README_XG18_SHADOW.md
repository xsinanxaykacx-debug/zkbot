# XG18 Shadow Production (research only)

This branch is isolated from `main` and the frozen model branch. It adds an append-only JSONL ledger and local CLI. It does not place bets, trigger decisions, tune parameters, or access the sealed final-test artifact.

## Frozen boundary

- Model: `zk2-xg18-frozen-ensemble-v1`; source commit `6053f2878cff4c0d73bc16ce65b2bb71ba26f742`.
- Frozen 1X2 weights: Team Poisson 0.13958888173927925, Dixon-Coles 0, Elo 0.8604111182607207.
- Team Poisson alpha 10; Elo K=32, home advantage=80, Davidson draw nu=0.8.
- The freeze manifest defines the ensemble as **1X2 only**. OU2.5 and BTTS are logged as auxiliary Team-Poisson-only diagnostics, not ensemble outputs.
- Team Poisson is always fit on the fixed 15,734 train+validation rows dated 2022-07-15 through 2026-03-14. Never add shadow outcomes to this fit file. Elo may replay settled shadow results from earlier UTC dates with unchanged Elo parameters. Same-day results are excluded from same-day forecasts.
- Never download, open, extract, read, score, or reuse the sealed final-test artifact.

## Prepare frozen source (Windows)

```cmd
research\assemble_xg18_source.cmd
powershell -NoProfile -Command "Expand-Archive -LiteralPath 'research\xg18-frozen-source.zip' -DestinationPath 'research\xg18-frozen-source' -Force"
```

Install the pinned packages from `research/xg18-frozen-source/requirements-xg18-freeze.txt` in an isolated environment.

## Input files

- `history.csv`: fixed 15,734 train+validation rows; columns include League, Date, HomeTeam, AwayTeam, FTHG, FTAG.
- `fixtures.csv`: upcoming fixtures; columns MatchID, League, Date, HomeTeam, AwayTeam. MatchID must be stable and unique. Date must be UTC. Fixtures must be future-dated and after fixed history.
- `odds.csv`: MatchID, captured_at, home_odds, draw_odds, away_odds, source, is_closing. Source must be Pinnacle; decimal odds >1; timestamp before kickoff. Mark is_closing=true only if the provider identifies the snapshot as the closing price. Fair probabilities use proportional normalization of inverse odds.
- `results.csv`: MatchID, FTHG, FTAG, result_available_at. Add results only after they are actually available.

Templates are in `research/shadow_*_template.csv`.

## Commands

```bash
python scripts/xg18_shadow.py predict --history data/xg18_train_validation.csv --fixtures data/upcoming_fixtures.csv --ledger shadow/shadow_ledger.jsonl
python scripts/xg18_shadow.py add-odds --input data/pinnacle_closing_odds.csv --ledger shadow/shadow_ledger.jsonl
python scripts/xg18_shadow.py settle --input data/settled_results.csv --ledger shadow/shadow_ledger.jsonl
python scripts/xg18_shadow.py report --ledger shadow/shadow_ledger.jsonl --output reports/xg18_shadow_report.json
```

Predictions, odds snapshots, and results are append-only events. Duplicate predictions, results, or closing prices are rejected. Keep the ledger local and backed up; do not hand-edit it. No Pinnacle scraping or API feed is included: the first release accepts timestamped CSV snapshots only.

## Reporting and gate

The report computes 1X2 Log Loss, multiclass Brier, accuracy, and 10-bin top-label ECE for ensemble and Elo; league-level ensemble metrics; and the same metrics for the ensemble versus verified Pinnacle closing fair probabilities on the overlapping subset. OU2.5/BTTS are auxiliary diagnostics only.

The observation gate requires **both** at least 1,000 settled matches and at least 56 calendar days between prediction timestamps. Passing the gate does not approve production; `production_approval` remains false. Shadow results are monitoring evidence only and must not be used to tune this frozen candidate.
