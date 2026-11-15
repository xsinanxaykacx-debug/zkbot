from zk2.data.team_mapping import normalize_team


def test_approved_aliases_map_to_canonical_names():
    assert normalize_team("Man United") == "Manchester United"
    assert normalize_team("MAN UTD") == "Manchester United"
    assert normalize_team("Galatasaray SK") == "Galatasaray"


def test_unknown_team_is_preserved():
    assert normalize_team("Bilinmeyen Takım") == "Bilinmeyen Takım"
