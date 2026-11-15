"""Explicit team aliases. Unknown names are preserved, never fuzzy-mapped."""

import re
import unicodedata
from typing import Mapping


def team_key(name: object) -> str:
    """Create a comparison key without deciding club identity."""
    if name is None:
        return ""
    value = unicodedata.normalize("NFKD", str(name).strip().casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("ı", "i")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


# This starter map contains only aliases that are sufficiently unambiguous in
# the intended competition context. Extend only after reviewing source evidence.
TEAM_CANONICAL: dict[str, str] = {
    "galatasaray": "Galatasaray",
    "galatasaray sk": "Galatasaray",
    "fenerbahce": "Fenerbahçe",
    "fenerbahce sk": "Fenerbahçe",
    "besiktas": "Beşiktaş",
    "besiktas jk": "Beşiktaş",
    "trabzonspor": "Trabzonspor",
    "caykur rizespor": "Çaykur Rizespor",
    "rizespor": "Çaykur Rizespor",
    "amed sportif": "Amed Sportif",
    "amedspor": "Amed Sportif",
    "eyupspor": "Eyüpspor",
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester utd": "Manchester United",
    "manchester united": "Manchester United",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "newcastle united": "Newcastle United",
    "nott m forest": "Nottingham Forest",
    "nottingham forest": "Nottingham Forest",
    "spurs": "Tottenham Hotspur",
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
}


def normalize_team(name: object, aliases: Mapping[str, str] | None = None) -> str:
    """Return approved canonical name; preserve unknown raw name unchanged."""
    raw = "" if name is None else str(name).strip()
    if not raw:
        return raw
    mapping = TEAM_CANONICAL if aliases is None else aliases
    return mapping.get(team_key(raw), raw)
