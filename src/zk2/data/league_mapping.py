"""Explicit competition identity mapping for known DATA.csv league labels.

Unknown labels are preserved verbatim; no fuzzy league merging is performed.
"""
from __future__ import annotations

import re


_LEAGUE_ALIASES = {
    "türkiye spor toto süper lig": "Türkiye Süper Lig",
    "türkiye trendyol süper lig": "Türkiye Süper Lig",
    "türkiye spor toto süper lig ahmet çalık sezonu": "Türkiye Süper Lig",
    "türkiye trendyol süper lig şamil ekinci sezonu": "Türkiye Süper Lig",
    "türkiye süper lig": "Türkiye Süper Lig",
    "türkiye tff 1. lig": "Türkiye 1. Lig",
    "türkiye tff 1.lig": "Türkiye 1. Lig",
    "türkiye spor toto 1. lig": "Türkiye 1. Lig",
    "türkiye trendyol 1. lig": "Türkiye 1. Lig",
    "türkiye ptt 1. lig": "Türkiye 1. Lig",
}


def normalize_league(value: object) -> object:
    """Map only explicitly reviewed aliases; leave missing/unknown values intact."""
    if value is None:
        return value
    try:
        if value != value:  # NaN / pandas.NA safe enough without importing pandas
            return value
    except (TypeError, ValueError):
        return value
    raw = str(value).strip()
    if not raw:
        return raw
    key = re.sub(r"\s+", " ", raw).casefold()
    return _LEAGUE_ALIASES.get(key, raw)
