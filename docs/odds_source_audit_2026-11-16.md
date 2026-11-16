# XG18 odds-source audit — 2026-11-16

## Turkey TFF 1. Lig (T2)

### Source candidate: CheckBestOdds
Public archive pages expose a distinct `Div 1` section with 1/X/2 prices for all target seasons checked:
- 2022/23: https://checkbestodds.com/football-odds/archive-turkey/2022-2023
- 2023/24: https://checkbestodds.com/football-odds/archive-turkey/2023-2024
- 2024/25: https://checkbestodds.com/football-odds/archive-turkey/2024-2025
- 2025/26: https://checkbestodds.com/football-odds/archive-turkey/2025-2026

The site describes displayed values as the "last highest odds we recorded for each of the bookmakers." This is not proven to be closing odds, and it is not equivalent to Football-Data's bookmaker-specific closing/average fields.

**Do not bulk extract or ingest yet.** The published Terms and Conditions describe informational odds services and subscriptions but do not explicitly grant a license for bulk reuse, redistribution, or machine-learning training. No public CSV/export was verified. Obtain written permission and confirm bookmaker/time semantics first. If licensed, retain these as a distinct source/feature family; do not map them into B365/average closing columns.

### Other sources checked
- OddsPortal has TFF 1. Lig historical pages and visible 1/X/2 prices, but no public bulk CSV/export was visible. Its terms prohibit automated access/scraping and substantial bulk extraction without permission; do not automate collection.
- Football-Data's Turkey page exposes T1/Futbol Ligi 1, not a separate TFF 1. Lig T2 odds CSV.
- FootyStats has TFF 1. Lig match/stat datasets, but no verified odds dataset.
- No T2 odds have been downloaded or added to the model.

## England League One missing odds

### 15 play-off matches
CheckBestOdds has a separate "League One Playoff" section in the England historical archives for 2022/23, 2023/24 and 2024/25. All 15 missing xG18 fixtures appear with 1/X/2 price rows. The source describes these as last highest odds recorded, not verified closing prices. Some fixtures show multiple rows with different listed times; bookmaker/time identity must be checked before selecting a price. No bulk-reuse license was confirmed; do not ingest yet.

The exact MatchIDs, dates, fixtures, scores, and classifications are in `data/league_one_missing_odds_manifest.csv`.

### Two false "missing match" records
These are abandoned events, not completed fixtures, and their 0-0 event records must not be treated as played matches:
- MatchID 4203250: Reading–Port Vale, 2024-01-13, abandoned after 16 minutes; actual replay MatchID 4417601 on 2024-02-20 is retained and has E2 odds.
- MatchID 4203025: Cambridge United–Bolton Wanderers, 2024-02-06, abandoned after 9 minutes; actual replay MatchID 4421571 on 2024-02-20 is retained and has E2 odds.

The confirmation workflow quarantines only the two abandoned records. The 15 completed play-off fixtures remain eligible for modelling, with odds missing until a permitted source is secured.

## Guardrails
- Work is isolated on `work/xg18-v3-confirmation`.
- No changes to `main` or production.
- Official final-test rows are never read.
- Do not import CheckBestOdds or OddsPortal values into the current odds schema until licensing, odds definition, bookmaker identity, and match coverage are verified.

### Licensed custom-data path: Footiqo
Footiqo's premium pre-match odds collection explicitly permits purchased datasets for personal research, statistical analysis, backtesting and predictive modelling, including ML; it also supports custom competition/season/bookmaker requests. Its standard odds collection currently lists eight competitions and does not list Turkey TFF 1. Lig. Therefore the practical licensed route is a custom quote/request for TFF 1. Lig seasons 2022/23–2025/26, with 1X2 and O/U 2.5 closing prices and bookmaker/timestamp fields. No order has been placed and no data have been received. Contact listed by the provider: support@footiqo.com.
