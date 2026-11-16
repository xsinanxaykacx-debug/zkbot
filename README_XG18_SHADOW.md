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
- `odds.csv`: MatchID, kickoff_utc, captured_at, home_odds, draw_odds, away_odds, source, provider_id, provider_event_id, pinnacle_source_id, snapshot_kind, is_closing. `source` is `pinnacle` only for the official Pinnacle API (`provider_id=pinnacle_official`); all independent services must use `source=third_party` and their own `provider_id` (for example `pinnapi` or `pinnodds`). `pinnacle_source_id` is optional and must never be fabricated. `captured_at` is accompanied by `timestamp_semantics`: it may be a provider-supplied timestamp of unverified meaning or local ingestion time. Decimal odds must be >1 and timestamps must precede kickoff. Only an official Pinnacle response may set `is_closing=true` for official closing metrics. Fair probabilities use proportional normalization of inverse odds.
- `results.csv`: MatchID, FTHG, FTAG, result_available_at. Add results only after they are actually available.

Templates are in `research/shadow_*_template.csv`. `research/shadow_odds_synthetic_16.csv` contains 16 explicitly synthetic rows across eight leagues for schema/manual dry-run checks only; it is not real provider evidence and must never be imported into the shadow ledger or used in model evaluation. An example configurable API field map is in `research/odds_api_mapping.example.json`; the Pinnodds and pinnapi event maps are `research/pinnodds_mapping.example.json` and `research/pinnapi_mapping.example.json` respectively. Their reviewed crosswalk templates are `research/pinnodds_match_map.template.csv` and `research/pinnapi_match_map.template.csv`. Use `research/pinnodds_match_map.template.csv` to build a manually reviewed, exact provider-event-ID to internal MatchID crosswalk. The adapter keeps `provider_event_id` separate from `pinnacle_source_id`: independent providers populate only `provider_event_id`; `pinnacle_source_id` is reserved for an actual official Pinnacle identifier. The adapter will not fuzzy-match or guess fixture identity. Provider league/team names and kickoff are checked against the shadow prediction before a snapshot is accepted.

## Commands

```bash
python scripts/xg18_shadow.py predict --history data/xg18_train_validation.csv --fixtures data/upcoming_fixtures.csv --ledger shadow/shadow_ledger.jsonl
python scripts/xg18_odds_adapter.py api --max-attempts 3 --backoff 1 --max-backoff 8 --url "https://pinnodds.com/kit/v1/prematch/fixtures?sport_id=1" --provider-id pinnodds --api-key-env PINNODDS_API_KEY --auth-header x-portal-apikey --auth-prefix "" --records-path events --mapping research/pinnodds_mapping.example.json --match-map research/pinnodds_match_map.csv --output data/normalized_odds.csv
python scripts/xg18_shadow.py add-odds --input data/normalized_odds.csv --ledger shadow/shadow_ledger.jsonl
python scripts/xg18_odds_adapter.py api --url "https://pinnapi.com/kit/v1/prematch/fixtures?sport_id=1" --provider-id pinnapi --api-key-env PINNAPI_API_KEY --auth-header x-portal-apikey --auth-prefix "" --records-path events --mapping research/pinnapi_mapping.example.json --match-map research/pinnapi_match_map.csv --output data/pinnapi_normalized_odds.csv
python scripts/xg18_shadow.py add-odds --input data/pinnapi_normalized_odds.csv --ledger shadow/shadow_ledger.jsonl
python scripts/xg18_shadow.py settle --input data/settled_results.csv --ledger shadow/shadow_ledger.jsonl
python scripts/xg18_shadow.py report --ledger shadow/shadow_ledger.jsonl --output reports/xg18_shadow_report.json
```

Predictions, odds snapshots, and results are append-only events. Duplicate predictions, results, or official closing prices are rejected. The adapter supports CSV normalization and generic HTTPS JSON GET/POST ingestion using a user-supplied field map; it does not contain provider credentials or a claim of provider verification. It validates required schema fields, explicit timezone-aware timestamps, kickoff ordering, decimal odds, provider/source identity, and closing-claim restrictions before writing normalized rows. Transient transport failures, HTTP 429 and HTTP 5xx receive bounded exponential retry (defaults: 3 attempts, 1s base, 8s cap); other 4xx responses fail immediately. Retry logs contain status/error type and delay only—never request headers or credentials. API keys are read only from environment variables and are never written to the ledger. Required normalized fields are MatchID, kickoff_utc, captured_at, home_odds, draw_odds, away_odds, source, provider_id, snapshot_kind, is_closing and timestamp_semantics; pinnacle_source_id and provider naming fields are optional metadata. Keep the ledger local and backed up; do not hand-edit it. No scraping or unauthorized endpoints are used. API records must first be normalized to the common CSV schema; that CSV is then consumed by `add-odds`.\n\n`research/pinnodds_mapping.example.json` maps the documented prematch event ID, kickoff, team names, league and full-match money line. Because the documented event payload does not expose a verified Pinnacle price-change timestamp, the adapter uses local ingestion time and labels it `local_ingestion_time`; do not misrepresent it as Pinnacle source time. The endpoint is prematch/live snapshot data, not a historical archive. `pinnapi` must use its own reviewed mapping file after a sample response is inspected.\n\n`observed_closing_proxy` is calculated separately in reports: the latest provider snapshot at or before kickoff, no more than 120 seconds old. It is explicitly labelled a proxy and is never counted as official Pinnacle closing. Independent providers remain unverified until real fixtures, timestamp semantics, odds, coverage, and licensing are checked.

## Reporting and gate

The report computes 1X2 Log Loss, multiclass Brier, accuracy, and 10-bin top-label ECE for ensemble and Elo; league-level ensemble metrics; and the same metrics for the ensemble versus verified Pinnacle closing fair probabilities on the overlapping subset. OU2.5/BTTS are auxiliary diagnostics only.

The observation gate requires **both** at least 1,000 settled matches and at least 56 calendar days between prediction timestamps. Passing the gate does not approve production; `production_approval` remains false. Shadow results are monitoring evidence only and must not be used to tune this frozen candidate.
