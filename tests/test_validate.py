import numpy as np
import pandas as pd

from zk2.data.validate import validate_schema


def test_reports_missing_required_columns_without_mutating_frame():
    frame = pd.DataFrame({"Tarih": ["2024-01-01"]})
    original = frame.copy(deep=True)

    report = validate_schema(frame, required_columns=["Tarih", "MSE", "MSD"])

    assert report.valid is False
    assert report.row_count == 1
    assert report.missing_required_columns == ["MSE", "MSD"]
    pd.testing.assert_frame_equal(frame, original)


def test_reports_empty_source_and_zero_fill_rates():
    report = validate_schema(
        pd.DataFrame(),
        required_columns=["Tarih"],
        inspect_columns=["MS1"],
        date_column="Tarih",
        key_columns=["match_id"],
    )

    assert report.row_count == 0
    assert report.valid is False
    assert report.column_fill_rates == {}
    assert report.warnings == ["Source contains zero rows."]


def test_calculates_fill_rates_and_invalid_date_count():
    frame = pd.DataFrame({
        "Tarih": ["2024-01-01", "invalid", "2024-01-03", None],
        "MS1": [1.5, 2.0, np.nan, 1.8],
    })

    report = validate_schema(
        frame,
        required_columns=["Tarih"],
        inspect_columns=["MS1"],
        date_column="Tarih",
    )

    assert report.valid is True
    assert report.column_fill_rates["Tarih"] == 0.75
    assert report.column_fill_rates["MS1"] == 0.75
    assert report.invalid_date_count == 2
    assert any("2 rows" in warning for warning in report.warnings)


def test_counts_all_rows_participating_in_duplicate_keys():
    frame = pd.DataFrame({
        "match_id": [1, 1, 2, 3, 3],
        "date": ["2024-01-01"] * 5,
    })

    report = validate_schema(
        frame,
        required_columns=["match_id", "date"],
        key_columns=["match_id", "date"],
    )

    assert report.duplicate_key_count == 4
    assert any("4 rows" in warning for warning in report.warnings)


def test_missing_key_columns_do_not_create_false_duplicate_count():
    frame = pd.DataFrame({"match_id": [1, 1]})

    report = validate_schema(
        frame,
        required_columns=["match_id"],
        key_columns=["match_id", "date"],
    )

    assert report.duplicate_key_count == 0


def test_valid_report_has_no_warnings_for_clean_source():
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "match_id": [1, 2],
        "odds": [1.8, 2.1],
    })

    report = validate_schema(
        frame,
        required_columns=["date", "match_id"],
        inspect_columns=["odds"],
        date_column="date",
        key_columns=["match_id"],
    )

    assert report.valid is True
    assert report.invalid_date_count == 0
    assert report.duplicate_key_count == 0
    assert report.column_fill_rates == {"date": 1.0, "match_id": 1.0, "odds": 1.0}
    assert report.warnings == []
