from zk2.data.team_mapping import normalize_team, team_key


def test_approved_aliases_map_to_canonical_names():
    assert normalize_team("Man United") == "Manchester United"
    assert normalize_team("MAN UTD") == "Manchester United"
    assert normalize_team("Galatasaray SK") == "Galatasaray"


def test_unknown_team_is_preserved():
    assert normalize_team("Bilinmeyen Takım") == "Bilinmeyen Takım"


def test_turkish_diacritics_are_normalized_for_comparison_only():
    assert team_key("FENERBAHÇE") == "fenerbahce"
    assert team_key("IĞDIR") == "igdir"
    # Ambiguous short alias is not globally canonicalized.
    assert normalize_team("Rizespor") == "Rizespor"


def test_reviewed_turkish_sponsor_and_renaming_aliases():
    assert normalize_team("Medipol Başakşehir") == "Başakşehir"
    assert normalize_team("Başakşehir FK") == "Başakşehir"
    assert normalize_team("Aytemiz Alanyaspor") == "Alanyaspor"
    assert normalize_team("Torku Konyaspor") == "Konyaspor"
    assert normalize_team("Amed SK") == "Amed Sportif"
    assert normalize_team("Bodrumspor") == "Bodrum FK"
    assert normalize_team("Gazişehir Gaziantep") == "Gaziantep FK"


def test_distinct_gaziantep_club_is_not_merged():
    assert normalize_team("Gaziantepspor") == "Gaziantepspor"
