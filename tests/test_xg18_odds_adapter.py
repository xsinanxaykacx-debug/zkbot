import unittest

from scripts.xg18_odds_adapter import normalize

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
            "h": 2.1,
            "d": 3.2,
            "a": 3.6,
        }

    def test_normalizes_third_party_with_utc(self):
        row = normalize(self.record, MAPPING, "pinnodds", "third_party")
        self.assertEqual(row["MatchID"], "fixture-1")
        self.assertEqual(row["provider_id"], "pinnodds")
        self.assertEqual(row["source"], "third_party")
        self.assertEqual(row["captured_at"], "2026-11-20T17:58:30Z")
        self.assertFalse(row["is_closing"])

    def test_rejects_naive_timestamp(self):
        record = dict(self.record, updated_at="2026-11-20 17:58:30")
        with self.assertRaises(ValueError):
            normalize(record, MAPPING, "pinnapi", "third_party")

    def test_rejects_non_official_pinnacle_source(self):
        with self.assertRaises(ValueError):
            normalize(self.record, MAPPING, "pinnapi", "pinnacle")

    def test_rejects_invalid_decimal_odds(self):
        record = dict(self.record, h=1.0)
        with self.assertRaises(ValueError):
            normalize(record, MAPPING, "pinnapi", "third_party")

if __name__ == "__main__":
    unittest.main()
