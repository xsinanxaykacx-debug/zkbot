from zk2.data.epl_team_mapping import normalize_epl_team


def test_epl_short_names_are_scoped_to_premier_league_identity():
    assert normalize_epl_team("Newcastle") == "Newcastle United"
    assert normalize_epl_team("Man United") == "Manchester United"
    assert normalize_epl_team("Nott'm Forest") == "Nottingham Forest"
    assert normalize_epl_team("Brighton") == "Brighton & Hove Albion"
    assert normalize_epl_team("West Brom") == "West Bromwich Albion"
    assert normalize_epl_team("Wolves") == "Wolverhampton Wanderers"


def test_unknown_epl_team_is_preserved():
    assert normalize_epl_team("Unknown Club") == "Unknown Club"


def test_empty_values_are_preserved_for_missingness_handling():
    assert normalize_epl_team("") == ""
    assert normalize_epl_team(None) == ""
