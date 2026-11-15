import pandas as pd

from zk2.data.league_mapping import normalize_league


def test_super_lig_sponsor_and_season_labels_share_identity():
    labels = [
        "Türkiye Spor Toto Süper Lig",
        "Türkiye Trendyol Süper Lig",
        "Türkiye Spor Toto Süper Lig Ahmet Çalık Sezonu",
        "Türkiye Trendyol Süper Lig Şamil Ekinci Sezonu",
        "Türkiye Süper Lig",
    ]
    assert {normalize_league(value) for value in labels} == {"Türkiye Süper Lig"}


def test_first_division_sponsor_and_spacing_aliases_share_identity():
    labels = [
        "Türkiye TFF 1. Lig",
        "Türkiye TFF 1.Lig",
        "Türkiye Spor Toto 1. Lig",
        "Türkiye Trendyol 1. Lig",
        "Türkiye PTT 1. Lig",
    ]
    assert {normalize_league(value) for value in labels} == {"Türkiye 1. Lig"}


def test_unknown_league_is_preserved_without_fuzzy_mapping():
    assert normalize_league("Unknown Regional Cup") == "Unknown Regional Cup"


def test_missing_league_values_remain_missing():
    assert pd.isna(normalize_league(float("nan")))
    assert normalize_league(None) is None
