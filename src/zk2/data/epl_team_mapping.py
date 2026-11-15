"""Competition-scoped aliases for the English Premier League workbook.

Short names are safe here because the source is explicitly EPL DATA, not a
worldwide mixed-competition feed. Unknown labels are preserved unchanged.
"""
from __future__ import annotations

from zk2.data.team_mapping import team_key

EPL_TEAM_CANONICAL: dict[str, str] = {
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "bournemouth": "AFC Bournemouth",
    "brighton": "Brighton & Hove Albion",
    "brentford": "Brentford",
    "burnley": "Burnley",
    "cardiff": "Cardiff City",
    "chelsea": "Chelsea",
    "crystal palace": "Crystal Palace",
    "coventry": "Coventry City",
    "everton": "Everton",
    "fulham": "Fulham",
    "huddersfield": "Huddersfield Town",
    "hull": "Hull City",
    "ipswich": "Ipswich Town",
    "leeds": "Leeds United",
    "leicester": "Leicester City",
    "liverpool": "Liverpool",
    "luton": "Luton Town",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester united": "Manchester United",
    "middlesbrough": "Middlesbrough",
    "newcastle": "Newcastle United",
    "newcastle united": "Newcastle United",
    "norwich": "Norwich City",
    "nott m forest": "Nottingham Forest",
    "nottingham forest": "Nottingham Forest",
    "southampton": "Southampton",
    "sheffield united": "Sheffield United",
    "stoke": "Stoke City",
    "sunderland": "Sunderland",
    "swansea": "Swansea City",
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    "watford": "Watford",
    "west brom": "West Bromwich Albion",
    "west bromwich albion": "West Bromwich Albion",
    "west ham": "West Ham United",
    "west ham united": "West Ham United",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
}


def normalize_epl_team(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return raw
    return EPL_TEAM_CANONICAL.get(team_key(raw), raw)
