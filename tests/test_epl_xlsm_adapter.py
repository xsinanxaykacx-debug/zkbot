from datetime import datetime

from openpyxl import Workbook

from zk2.data.adapters.epl_xlsm import EplXlsmAdapter


def _write_book(path):
    workbook = Workbook()
    data = workbook.active
    data.title = "DATA"
    data.append([
        "Tarih", "EvSahibi", "Deplasman", "MS_EvGol", "MS_DepGol",
        "Bet365_Ev", "Bet365_Beraberlik", "Bet365_Dep",
        "EnYüksekEvOranı", "EnYüksekEvOranı", "Saat", "Bet365_Üst2,5",
    ])
    data.append(["08/08/2015", "Man United", "Tottenham", 2, 1, 1.8, 3.5, 4.2, 1.75, 1.77, "15:00", 1.9])
    data.append(["13/08/16", "Newcastle", "Brighton", 0, 0, 2.1, 3.2, 3.5, 2.0, 2.02, "17:30", 2.1])
    data.append(["bad-date", "Unknown Club", "Arsenal", None, None, None, None, None, None, None, None, None])
    report = workbook.create_sheet("RAPOR")
    report.append(["this sheet must not be loaded"])
    workbook.save(path)
    return path


def test_epl_adapter_reads_only_data_and_preserves_duplicate_headers(tmp_path):
    path = _write_book(tmp_path / "epl.xlsm")
    result = EplXlsmAdapter(path).load().transform()
    assert len(result) == 3
    assert "EnYüksekEvOranı__col_9" in result.columns
    assert "EnYüksekEvOranı__col_10" in result.columns
    assert result.loc[0, "_team_home_canonical"] == "Manchester United"
    assert result.loc[0, "_team_away_canonical"] == "Tottenham Hotspur"
    assert result.loc[1, "_team_home_canonical"] == "Newcastle United"
    assert result.loc[1, "_team_away_canonical"] == "Brighton & Hove Albion"
    assert result.loc[0, "_league_canonical"] == "England Premier League"


def test_epl_adapter_builds_targets_and_preserves_missing_scores(tmp_path):
    result = EplXlsmAdapter(_write_book(tmp_path / "epl.xlsm")).load().transform()
    assert result.loc[0, "_y_1x2"] == "H"
    assert result.loc[0, "_y_ou25"] == 1
    assert result.loc[0, "_y_kg"] == 1
    assert result.loc[2, "_y_1x2"] is None or str(result.loc[2, "_y_1x2"]) == "<NA>"
    assert result.loc[2, "_y_ou25"] is None or str(result.loc[2, "_y_ou25"]) == "<NA>"


def test_epl_adapter_parses_two_digit_year_and_reports_invalid_dates(tmp_path):
    result = EplXlsmAdapter(_write_book(tmp_path / "epl.xlsm")).load().transform()
    parsed = result.loc[result["_source_row"] == 3].iloc[0]
    assert parsed["_date"] == datetime(2016, 8, 13)
    assert bool(parsed["_is_valid_date"]) is True
    assert bool(result.loc[result["_source_row"] == 4, "_is_valid_date"].iloc[0]) is False


def test_epl_adapter_audit_keeps_all_rows(tmp_path):
    result = EplXlsmAdapter(_write_book(tmp_path / "epl.xlsm")).load().transform()
    report = EplXlsmAdapter.audit(result)
    assert report["rows"] == 3
    assert report["invalid_date_rows"] == 1
    assert report["missing_score_rows"] == 1
