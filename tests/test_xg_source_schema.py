import pandas as pd
from openpyxl import Workbook

from scripts.audit_xg_source_schema import audit_sources, is_xg_header


def test_xg_header_detection_is_specific():
    assert is_xg_header("Home xG")
    assert is_xg_header("expected_goals")
    assert is_xg_header("Away Expected Goals")
    assert not is_xg_header("xgboost_version")
    assert not is_xg_header("Goals")


def test_audit_sources_reports_match_level_column_coverage(tmp_path):
    csv_path = tmp_path / "DATA.csv"
    pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Home xG": [1.2, None, 0.7],
        "Away xG": [0.4, 1.1, None],
        "Goals": [2, 0, 1],
    }).to_csv(csv_path, index=False)

    xlsm_path = tmp_path / "EPL.xlsm"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "DATA"
    sheet.append(["Tarih", "EvSahibi", "Expected Goals Home", "Expected Goals Away"])
    sheet.append(["01/01/2024", "A", 1.3, 0.5])
    sheet.append(["02/01/2024", "B", None, 1.0])
    workbook.save(xlsm_path)

    result = audit_sources(csv_path, xlsm_path)
    csv_source, xlsm_source = result["sources"]

    assert csv_source["xg_candidate_column_count"] == 2
    assert csv_source["xg_candidate_columns"][0]["non_null_rows"] == 2
    assert csv_source["xg_candidate_columns"][0]["numeric_rows"] == 2
    assert xlsm_source["xg_candidate_column_count"] == 2
    assert xlsm_source["xg_candidate_columns"][0]["non_null_rows"] == 1
    assert "FotMob" in result["purpose"]
