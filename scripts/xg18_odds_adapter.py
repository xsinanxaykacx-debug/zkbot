#!/usr/bin/env python3
"""Provider-neutral odds ingestion for XG18 shadow. No provider is trusted by default."""
import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "MatchID", "kickoff_utc", "captured_at", "home_odds", "draw_odds",
    "away_odds", "source", "provider_id", "pinnacle_source_id",
    "snapshot_kind", "is_closing", "provider_league", "provider_home_team",
    "provider_away_team", "timestamp_semantics",
]

def value_at(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur

def utc_iso(value):
    from datetime import datetime
    v = str(value).strip()
    if not v:
        raise ValueError("empty timestamp")
    d = datetime.fromisoformat(v.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("timestamps must include timezone; naive local time is rejected")
    return d.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def normalize(item, mapping, provider_id, source):
    row = {}
    for field in ("MatchID", "kickoff_utc", "home_odds", "draw_odds", "away_odds"):
        row[field] = value_at(item, mapping[field])
    row["MatchID"] = str(row["MatchID"]).strip()
    if not row["MatchID"]:
        raise ValueError("empty MatchID")
    row["kickoff_utc"] = utc_iso(row["kickoff_utc"])
    if mapping["captured_at"] == "$ingested_at":
        row["captured_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        row["timestamp_semantics"] = "local_ingestion_time"
    else:
        row["captured_at"] = utc_iso(row["captured_at"])
        row["timestamp_semantics"] = "provider_supplied_timestamp_unverified"
    for field in ("provider_league", "provider_home_team", "provider_away_team"):
        path = mapping.get(field)
        row[field] = str(value_at(item, path)) if path else ""
    for field in ("home_odds", "draw_odds", "away_odds"):
        row[field] = float(row[field])
        if not (row[field] > 1 and row[field] < float("inf")):
            raise ValueError(f"{field} must be finite decimal odds > 1")
    if source == "pinnacle" and provider_id != "pinnacle_official":
        raise ValueError("Only provider_id=pinnacle_official may use source=pinnacle")
    row["source"] = source
    row["provider_id"] = provider_id
    p = mapping.get("pinnacle_source_id")
    row["pinnacle_source_id"] = str(value_at(item, p)) if p else ""
    k = mapping.get("snapshot_kind")
    row["snapshot_kind"] = str(value_at(item, k)) if k else "live_snapshot"
    c = mapping.get("is_closing")
    row["is_closing"] = str(value_at(item, c)).lower() in ("true", "1", "yes") if c else False
    return row

def apply_match_map(rows, path):
    if not path:
        return rows
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    required = {"provider_event_id", "MatchID"}
    if not records or not required.issubset(records[0]):
        raise ValueError("match-map CSV requires provider_event_id,MatchID")
    mapping = {}
    targets = set()
    for r in records:
        key = str(r["provider_event_id"]).strip()
        target = str(r["MatchID"]).strip()
        if not key or not target or key in mapping or target in targets:
            raise ValueError("match-map must be non-empty and one-to-one")
        mapping[key] = target
        targets.add(target)
    for row in rows:
        key = str(row.get("pinnacle_source_id", "")).strip()
        if key not in mapping:
            raise ValueError(f"No exact reviewed match-map entry for provider event {key}")
        row["MatchID"] = mapping[key]
    return rows

def write_rows(rows, output):
    if not rows:
        raise ValueError("API/CSV returned zero odds records")
    seen = set()
    with Path(output).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            key = (row["MatchID"], row["captured_at"], row["provider_id"])
            if key in seen:
                raise ValueError(f"duplicate snapshot: {key}")
            seen.add(key)
            w.writerow(row)

def api_command(a):
    token = os.environ.get(a.api_key_env) if a.api_key_env else None
    headers = {"Accept": "application/json", "User-Agent": "XG18-shadow-research/1.0"}
    if a.api_key_env:
        if not token:
            raise ValueError(f"environment variable {a.api_key_env} is not set")
        headers[a.auth_header] = f"{a.auth_prefix}{token}"
    body = None
    if a.body_json:
        raw = json.loads(Path(a.body_json).read_text(encoding="utf-8"))
        if a.body_format == "form":
            from urllib.parse import urlencode
            body = urlencode(raw, doseq=True).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            body = json.dumps(raw).encode("utf-8")
            headers["Content-Type"] = "application/json"
    req = urllib.request.Request(a.url, data=body, headers=headers, method=a.method)
    with urllib.request.urlopen(req, timeout=a.timeout) as response:
        if response.status != 200:
            raise ValueError(f"provider HTTP status {response.status}")
        payload = json.loads(response.read().decode("utf-8"))
    records = value_at(payload, a.records_path) if a.records_path else payload
    if not isinstance(records, list):
        raise ValueError("records-path must resolve to a JSON array")
    mapping = json.loads(Path(a.mapping).read_text(encoding="utf-8"))
    rows = [normalize(x, mapping, a.provider_id, a.source) for x in records]
    write_rows(rows, a.output)
    print(json.dumps({"status":"normalized_not_verified","provider_id":a.provider_id,
                      "records":len(rows),"output":a.output,"credentials_logged":False}, indent=2))

def csv_command(a):
    mapping = json.loads(Path(a.mapping).read_text(encoding="utf-8"))
    rows = []
    with Path(a.input).open(newline="", encoding="utf-8-sig") as f:
        for item in csv.DictReader(f):
            rows.append(normalize(item, mapping, a.provider_id, a.source))
    write_rows(rows, a.output)
    print(json.dumps({"status":"normalized_not_verified","provider_id":a.provider_id,
                      "records":len(rows),"output":a.output}, indent=2))

def main():
    p = argparse.ArgumentParser(description="Normalize provider odds into the common XG18 CSV schema")
    s = p.add_subparsers(dest="command", required=True)
    q = s.add_parser("api")
    q.add_argument("--url", required=True)
    q.add_argument("--provider-id", required=True)
    q.add_argument("--source", choices=("pinnacle","third_party"), default="third_party")
    q.add_argument("--api-key-env", default="")
    q.add_argument("--auth-header", default="Authorization")
    q.add_argument("--auth-prefix", default="Bearer ")\n    q.add_argument("--method", choices=("GET", "POST"), default="GET")\n    q.add_argument("--body-json", default="", help="Optional JSON file containing POST body fields")\n    q.add_argument("--body-format", choices=("json", "form"), default="json")
    q.add_argument("--records-path", default="")
    q.add_argument("--mapping", required=True, help="JSON field-path mapping; no secrets")\n    q.add_argument("--match-map", default="", help="Reviewed exact provider_event_id to internal MatchID crosswalk CSV")
    q.add_argument("--output", required=True)
    q.add_argument("--timeout", type=int, default=20)
    q.set_defaults(fn=api_command)
    q = s.add_parser("csv")
    q.add_argument("--input", required=True)
    q.add_argument("--provider-id", required=True)
    q.add_argument("--source", choices=("pinnacle","third_party"), default="third_party")
    q.add_argument("--mapping", required=True)
    q.add_argument("--output", required=True)
    q.set_defaults(fn=csv_command)
    a = p.parse_args()
    try:
        a.fn(a)
    except (KeyError, IndexError, ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
