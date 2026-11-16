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
- `odds.csv`: MatchID, kickoff_utc, captured_at, home_odds, draw_odds, away_odds, source, provider_id, pinnacle_source_id, snapshot_kind, is_closing. `source` is `pinnacle` only for the official Pinnacle API (`provider_id=pinnacle_official`); all independent services must use `source=third_party` and their own `provider_id` (for example `pinnapi` or `pinnodds`). `pinnacle_source_id` is optional and must never be fabricated. `captured_at` is the timestamp supplied by that provider, not necessarily Pinnacle event time. Decimal odds must be >1 and timestamps must precede kickoff. Only an official Pinnacle response may set `is_closing=true` for official closing metrics. Fair probabilities use proportional normalization of inverse odds.
- `results.csv`: MatchID, FTHG, FTAG, result_available_at. Add results only after they are actually available.

Templates are in `research/shadow_*_template.csv`. An example configurable API field map is in `research/odds_api_mapping.example.json`; adapt it to the provider documentation before use.

## Commands

```bash
python scripts/xg18_shadow.py predict --history data/xg18_train_validation.csv --fixtures data/upcoming_fixtures.csv --ledger shadow/shadow_ledger.jsonl
python scripts/xg18_odds_adapter.py api --url "$ODDS_API_URL" --provider-id pinnapi --api-key-env PINNAPI_API_KEY --records-path data --mapping research/provider_mapping.json --output data/normalized_odds.csv\npython scripts/xg18_shadow.py add-odds --input data/normalized_odds.csv --ledger shadow/shadow_ledger.jsonl
python scripts/xg18_shadow.py settle --input data/settled_results.csv --ledger shadow/shadow_ledger.jsonl
python scripts/xg18_shadow.py report --ledger shadow/shadow_ledger.jsonl --output reports/xg18_shadow_report.json
```

Predictions, odds snapshots, and results are append-only events. Duplicate predictions, results, or official closing prices are rejected. The adapter supports CSV normalization and generic HTTPS JSON GET ingestion using a user-supplied field map; it does not contain provider-specific endpoints, credentials, or a claim of provider verification. API keys are read only from environment variables and are never written to the ledger. Keep the ledger local and backed up; do not hand-edit it. No scraping or unauthorized endpoints are used. API records must first be normalized to the common CSV schema; that CSV is then consumed by `add-odds`.\n\n`observed_closing_proxy` is calculated separately in reports: the latest provider snapshot at or before kickoff, no more than 120 seconds old. It is explicitly labelled a proxy and is never counted as official Pinnacle closing. Independent providers remain unverified until real fixtures, timestamp semantics, odds, coverage, and licensing are checked.

## Reporting and gate

The report computes 1X2 Log Loss, multiclass Brier, accuracy, and 10-bin top-label ECE for ensemble and Elo; league-level ensemble metrics; and the same metrics for the ensemble versus verified Pinnacle closing fair probabilities on the overlapping subset. OU2.5/BTTS are auxiliary diagnostics only.

The observation gate requires **both** at least 1,000 settled matches and at least 56 calendar days between prediction timestamps. Passing the gate does not approve production; `production_approval` remains false. Shadow results are monitoring evidence only and must not be used to tune this frozen candidate.
