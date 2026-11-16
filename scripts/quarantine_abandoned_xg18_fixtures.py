#!/usr/bin/env python3
"""Remove abandoned, non-played fixtures that xG18 records as 0-0."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

ABANDONED={
    "4203250":{"date":"2024-01-13","home":"Reading","away":"Port Vale","reason":"Abandoned after 16 minutes; replayed 2024-02-20"},
    "4203025":{"date":"2024-02-06","home":"Cambridge United","away":"Bolton Wanderers","reason":"Abandoned after 9 minutes; replayed 2024-02-20"},
}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    a=ap.parse_args()
    d=pd.read_csv(a.input,low_memory=False)
    if "MatchID" not in d:raise ValueError("MatchID missing")
    ids=d.MatchID.astype(str)
    found=d.loc[ids.isin(ABANDONED)].copy()
    if set(found.MatchID.astype(str))!=set(ABANDONED):
        raise ValueError(f"Expected exactly both abandoned IDs; found {found.MatchID.astype(str).tolist()}")
    for r in found.itertuples():
        meta=ABANDONED[str(r.MatchID)]
        if str(r.HomeTeam)!=meta["home"] or str(r.AwayTeam)!=meta["away"]:
            raise ValueError(f"Fixture identity mismatch for {r.MatchID}")
        if int(r.FTHG)!=0 or int(r.FTAG)!=0:
            raise ValueError(f"Unexpected abandoned fixture score for {r.MatchID}")
    clean=d.loc[~ids.isin(ABANDONED)].copy()
    if len(clean)!=len(d)-2:raise RuntimeError("Quarantine row count mismatch")
    a.output.parent.mkdir(parents=True,exist_ok=True);a.report.parent.mkdir(parents=True,exist_ok=True)
    clean.to_csv(a.output,index=False)
    report={"input_rows":len(d),"quarantined_rows":len(found),"output_rows":len(clean),
      "quarantined":[{"MatchID":mid,**meta} for mid,meta in ABANDONED.items()],
      "replay_fixtures_retained":True,
      "policy":"Only the two abandoned 0-0 event records are excluded; completed 2024-02-20 replay fixtures remain. No final-test rows are read."}
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
