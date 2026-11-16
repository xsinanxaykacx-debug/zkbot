import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_xg18_all_league_odds.py"
SPEC = importlib.util.spec_from_file_location("merge_xg18_all_league_odds", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_league_one_odds_aliases():
    cases = {
        "Wycombe Wanderers": "wycombe",
        "Derby County": "derby",
        "Oxford United": "oxford",
        "Bolton Wanderers": "bolton",
        "Peterborough United": "peterboro",
        "Accrington Stanley": "accrington",
        "Charlton Athletic": "charlton",
        "Bristol Rovers": "bristol rvs",
        "Forest Green Rovers": "forest green",
        "Cambridge United": "cambridge",
        "Ipswich Town": "ipswich",
        "Lincoln City": "lincoln",
        "Exeter City": "exeter",
    }
    for source, expected in cases.items():
        assert MODULE.team(source) == expected


def test_germany_2_bundesliga_odds_aliases():
    cases = {
        "Hannover 96": "hannover",
        "Jahn Regensburg": "regensburg",
        "Arminia Bielefeld": "bielefeld",
        "1. FC Nürnberg": "nurnberg",
        "Eintracht Braunschweig": "braunschweig",
        "Hamburger SV": "hamburg",
        "Karlsruher SC": "karlsruhe",
        "SC Paderborn 07": "paderborn",
        "St. Pauli": "st pauli",
    }
    for source, expected in cases.items():
        assert MODULE.team(source) == expected


def test_alias_matching_preserves_non_alias_normalization():
    assert MODULE.team("Manchester United") == MODULE.team("Man United")
    assert MODULE.team("St. Pauli") == MODULE.team("St Pauli")
