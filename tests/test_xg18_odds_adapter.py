import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.xg18_odds_adapter import (
    apply_match_map,
    normalize,
    validate_normalized_row,
)

MAPPING = {
    "MatchID": "id",
    "kickoff_utc": "kickoff",
    "captured_at": "updated_at",
    "home_odds": "h",
    "draw_odds": "d",
    "away_odds": "a",
}

class OddsAdapterTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "id": "fixture-1",
            "kickoff": "2026-11-20T18:00:00Z",
            "updated_at": "2026-11-20T17:58:30Z",
            "h": 2.1, "d": 3.2, "a": 3.6,
        }

    def test_normalizes_third_party_with_utc(self):
        row = normalize(self.record, MAPPING, "pinnodds", "third_party")
        self.assertEqual(row["MatchID"], "fixture-1")
        self.assertEqual(row["provider_id"], "pinnodds")
        self.assertEqual(row["source"], "third_party")
        self.assertEqual(row["captured_at"], "2026-11-20T17:58:30Z")
        self.assertFalse(row["is_closing"])

    def test_timezone_offset_is_normalized_to_utc(self):
        record = dict(self.record, updated_at="2026-11-20T20:58:30+03:00")
        row = normalize(record, MAPPING, "pinnapi", "third_party")
        self.assertEqual(row["captured_at"], "2026-11-20T17:58:30Z")

    def test_rejects_naive_timestamp(self):
        for value in ("2026-11-20 17:58:30", "", "not-a-date"):
            with self.subTest(value=value):
                record = dict(self.record, updated_at=value)
                with self.assertRaises((ValueError, TypeError)):
                    normalize(record, MAPPING, "pinnapi", "third_party")

    def test_rejects_naive_kickoff(self):
        record = dict(self.record, kickoff="2026-11-20T18:00:00")
        with self.assertRaises(ValueError):
            normalize(record, MAPPING, "pinnapi", "third_party")

    def test_rejects_capture_at_kickoff(self):
        record = dict(self.record, updated_at=self.record["kickoff"])
        with self.assertRaises(ValueError):
            normalize(record, MAPPING, "pinnapi", "third_party")

    def test_rejects_capture_after_kickoff(self):
        record = dict(self.record, updated_at="2026-11-20T18:00:01Z")
        with self.assertRaises(ValueError):
            normalize(record, MAPPING, "pinnapi", "third_party")

    def test_rejects_non_official_pinnacle_source(self):
        with self.assertRaises(ValueError):
            normalize(self.record, MAPPING, "pinnapi", "pinnacle")

    def test_rejects_official_id_mislabelled_third_party(self):
        with self.assertRaises(ValueError):
            normalize(self.record, MAPPING, "pinnacle_official", "third_party")

    def test_accepts_official_pinnacle_identity(self):
        row = normalize(self.record, MAPPING, "pinnacle_official", "pinnacle")
        self.assertEqual(row["source"], "pinnacle")
        self.assertEqual(row["provider_id"], "pinnacle_official")

    def test_rejects_invalid_decimal_odds(self):
        for value in (1, 0, -2, float("nan"), float("inf"), "not-number"):
            with self.subTest(value=value):
                record = dict(self.record, h=value)
                with self.assertRaises((ValueError, TypeError)):
                    normalize(record, MAPPING, "pinnapi", "third_party")

    def test_rejects_missing_required_field_mapping(self):
        mapping = dict(MAPPING)
        del mapping["draw_odds"]
        with self.assertRaises(KeyError):
            normalize(self.record, mapping, "pinnapi", "third_party")

    def test_rejects_empty_provider_event_id(self):
        record = dict(self.record, id=" ")
        with self.assertRaises(ValueError):
            normalize(record, MAPPING, "pinnapi", "third_party")

    def test_rejects_third_party_official_closing_claim(self):
        mapping = dict(MAPPING, is_closing="closing")
        record = dict(self.record, closing=True)
        with self.assertRaises(ValueError):
            normalize(record, mapping, "pinnodds", "third_party")

    def test_optional_provider_metadata_can_be_absent(self):
        row = normalize(self.record, MAPPING, "pinnodds", "third_party")
        self.assertEqual(row["provider_league"], "")
        self.assertEqual(row["provider_home_team"], "")

    def test_match_map_requires_exact_one_to_one_ids(self):
        rows = [{"pinnacle_source_id": "event-a"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.csv"
            path.write_text(
                "provider_event_id,MatchID\nevent-a,match-a\n",
                encoding="utf-8",
            )
            result = apply_match_map(rows, str(path))
            self.assertEqual(result[0]["MatchID"], "match-a")

    def test_match_map_rejects_duplicate_provider_ids(self):
        rows = [{"pinnacle_source_id": "event-a"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.csv"
            path.write_text(
                "provider_event_id,MatchID\nevent-a,match-a\nevent-a,match-b\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                apply_match_map(rows, str(path))

    def test_match_map_rejects_duplicate_internal_ids(self):
        rows = [{"pinnacle_source_id": "event-a"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.csv"
            path.write_text(
                "provider_event_id,MatchID\nevent-a,match-a\nevent-b,match-a\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                apply_match_map(rows, str(path))

    def test_match_map_rejects_unmapped_event(self):
        rows = [{"pinnacle_source_id": "unknown-event"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.csv"
            path.write_text(
                "provider_event_id,MatchID\nevent-a,match-a\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                apply_match_map(rows, str(path))

    def test_common_schema_validator_rejects_bad_provider(self):
        row = normalize(self.record, MAPPING, "pinnodds", "third_party")
        row["provider_id"] = ""
        with self.assertRaises(ValueError):
            validate_normalized_row(row)

    def test_common_schema_validator_rejects_missing_timestamp_semantics(self):
        row = normalize(self.record, MAPPING, "pinnodds", "third_party")
        row["timestamp_semantics"] = ""
        with self.assertRaises(ValueError):
            validate_normalized_row(row)

    def test_closing_flag_is_false_by_default(self):
        row = normalize(self.record, MAPPING, "pinnodds", "third_party")
        self.assertFalse(row["is_closing"])

if __name__ == "__main__":
    unittest.main()
