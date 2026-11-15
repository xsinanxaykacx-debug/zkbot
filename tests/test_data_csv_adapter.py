from pathlib import Path

import pandas as pd
import pytest

from zk2.data.adapters.data_csv import DataCsvAdapter, load_ou25_odds


BASE_ROW = {
    "Tarih": "11-01-2014",
    "Saat": "14:00",
    "Lig/Sezon": "Türkiye PTT 1. Lig 2013/2014",
    "Maç_ID": "1205783",
    "Ev_Sahibi": "Galatasaray SK",
    "Deplasman": "Fenerbahçe",
    "MSE": "1",
    "MSD": "3",
    "Maç_Sonucu_|_1": "1,50",
    "Maç_Sonucu_|_X": "3,40",
    "Maç_Sonucu_|_2": "4,20",
    "Karşılıklı_Gol_|_Var": "1,70",
    "Karşılıklı_Gol_|_Yok": "2,00",
    "AÜ_2,5_Gol_|_Alt": "1,80",
    "AÜ_2,5_Gol_|_Üst": "2,00",
    "2,5_Alt/Üst_|_Alt": None,
    "2,5_Alt/Üst_|_Üst": None,
}


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "DATA.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_adapter_loads_utf8_csv_and_transforms_targets(tmp_path):
    path = _write_csv(tmp_path, [BASE_ROW])
    result = DataCsvAdapter(path).load().transform()

    assert len(result) == 1
    assert result.loc[0, "_y_1x2"] == "A"
    assert result.loc[0, "_y_ou25"] == 1
    assert result.loc[0, "_y_kg"] == 1
    assert result.loc[0, "_team_home_canonical"] == "Galatasaray"
    assert result.loc[0, "_team_away_canonical"] == "Fenerbahçe"
    assert result.loc[0, "_league_canonical"] == "Türkiye PTT 1. Lig"
    assert result.loc[0, "_season"] == "2013/2014"
    assert result.loc[0, "_odd_H"] == pytest.approx(1.5)


def test_ou25_uses_legacy_first_and_new_as_row_fallback():
    frame = pd.DataFrame({
        "AÜ_2,5_Gol_|_Alt": ["1,80", None],
        "AÜ_2,5_Gol_|_Üst": ["2,00", None],
        "2,5_Alt/Üst_|_Alt": ["1,90", "1,95"],
        "2,5_Alt/Üst_|_Üst": ["1,95", "2,05"],
    })

    under, over = load_ou25_odds(frame)
    assert under.tolist() == pytest.approx([1.8, 1.95])
    assert over.tolist() == pytest.approx([2.0, 2.05])


def test_missing_scores_remain_missing_targets(tmp_path):
    row = {**BASE_ROW, "MSE": None, "MSD": "2"}
    result = DataCsvAdapter(_write_csv(tmp_path, [row])).load().transform()

    assert pd.isna(result.loc[0, "_y_1x2"])
    assert pd.isna(result.loc[0, "_y_ou25"])
    assert pd.isna(result.loc[0, "_y_kg"])


def test_invalid_date_and_season_rows_are_preserved_for_audit(tmp_path):
    row = {**BASE_ROW, "Tarih": "not-a-date", "Lig/Sezon": "Unknown League"}
    result = DataCsvAdapter(_write_csv(tmp_path, [row])).load().transform()

    assert len(result) == 1
    report = DataCsvAdapter.audit(result)
    assert report["rows"] == 1
    assert report["invalid_date_rows"] == 1
    assert report["missing_season_rows"] == 1


def test_transform_requires_source_schema_columns():
    adapter = DataCsvAdapter("unused.csv")
    with pytest.raises(ValueError, match="missing required columns"):
        adapter.transform(pd.DataFrame({"Tarih": ["11-01-2014"]}))


def test_ou25_rejects_incomplete_column_pair():
    with pytest.raises(ValueError, match="pair is incomplete"):
        load_ou25_odds(pd.DataFrame({"AÜ_2,5_Gol_|_Alt": ["1,8"]}))


def test_audit_reports_missing_odds_without_filling_them(tmp_path):
    row = {**BASE_ROW, "Maç_Sonucu_|_1": None, "Maç_Sonucu_|_X": None,
           "Maç_Sonucu_|_2": None}
    result = DataCsvAdapter(_write_csv(tmp_path, [row])).load().transform()
    report = DataCsvAdapter.audit(result)

    assert report["missing_1x2_odds_rows"] == 1
    assert pd.isna(result.loc[0, "_odd_H"])
