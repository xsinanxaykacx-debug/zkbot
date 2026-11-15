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


# Global aliases are restricted to names considered unambiguous. Short or
# competition-dependent aliases (for example "Rizespor") must be scoped later.
TEAM_CANONICAL: dict[str, str] = {
    "galatasaray": "Galatasaray",
    "galatasaray sk": "Galatasaray",
    "fenerbahce": "Fenerbahçe",
    "fenerbahce sk": "Fenerbahçe",
    "besiktas": "Beşiktaş",
    "besiktas jk": "Beşiktaş",
    "trabzonspor": "Trabzonspor",
    "caykur rizespor": "Çaykur Rizespor",
    "amed sportif": "Amed Sportif",
    "amedspor": "Amed Sportif",
    "eyupspor": "Eyüpspor",
    "amed sk": "Amed Sportif",
    "medipol basaksehir": "Başakşehir",
    "istanbul basaksehir": "Başakşehir",
    "basaksehir fk": "Başakşehir",
    "aytemiz alanyaspor": "Alanyaspor",
    "albimo alanyaspor": "Alanyaspor",
    "atiker konyaspor": "Konyaspor",
    "torku konyaspor": "Konyaspor",
    "akhisar belediyespor": "Akhisarspor",
    "balikesirspor baltok": "Balıkesirspor",
    "kardemir karabukspor": "Karabükspor",
    "tekden denizlispor": "Denizlispor",
    "medicana sivasspor": "Sivasspor",
    "mp antalyaspor": "Antalyaspor",
    "bb erzurumspor": "Erzurumspor",
    "erzurumspor fk": "Erzurumspor",
    "bodrumspor": "Bodrum FK",
    "bodrum fk": "Bodrum FK",
    "osmanlispor fk": "Ankaraspor",
    "ankaraspor": "Ankaraspor",
    "gaziantep bbsk": "Gaziantep FK",
    "buyuksehir gaziantepspor": "Gaziantep FK",
    "gazisehir gaziantep": "Gaziantep FK",
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
