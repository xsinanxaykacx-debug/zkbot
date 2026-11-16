#!/usr/bin/env python3
"""Attach only pre-match odds columns to the official XG18 train/validation split."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--merged",type=Path,required=True)
    ap.add_argument("--train",type=Path,required=True)
    ap.add_argument("--validation",type=Path,required=True)
    ap.add_argument("--outdir",type=Path,required=True)
    a=ap.parse_args()
    merged=pd.read_csv(a.merged,low_memory=False)
    if merged["MatchID"].astype(str).duplicated().any():
        raise ValueError("Merged odds dataset contains duplicate MatchIDs")
    odds_cols=[c for c in merged.columns if c.startswith("odds_")]
    if not odds_cols: raise ValueError("No odds columns found in merged dataset")
    lookup=merged[["MatchID",*odds_cols]].copy()
    lookup["MatchID"]=lookup["MatchID"].astype(str)
    outputs={}
    for label,path in (("train",a.train),("validation",a.validation)):
        base=pd.read_csv(path,low_memory=False)
        ids=base["MatchID"].astype(str)
        if ids.duplicated().any(): raise ValueError(f"{label} contains duplicate MatchIDs")
        joined=base.copy()
        joined["MatchID"]=ids
        joined=joined.merge(lookup,on="MatchID",how="left",validate="one_to_one",sort=False,indicator=True)
        missing=joined["_merge"]!="both"
        if missing.any(): raise ValueError(f"{label}: {int(missing.sum())} MatchIDs missing from merged odds dataset")
        joined=joined.drop(columns=["_merge"])
        if len(joined)!=len(base): raise ValueError(f"{label} row count changed during odds join")
        outputs[label]=joined
    a.outdir.mkdir(parents=True,exist_ok=True)
    for label,frame in outputs.items():
        frame.to_csv(a.outdir/f"xg18_{label}_odds.csv",index=False)
    print({"train_rows":len(outputs["train"]),"validation_rows":len(outputs["validation"]),
           "odds_columns":len(odds_cols),"train_matched_odds":int(outputs["train"]["odds_match_status"].astype(str).str.startswith("matched").sum()),
           "validation_matched_odds":int(outputs["validation"]["odds_match_status"].astype(str).str.startswith("matched").sum())})
if __name__=="__main__": main()
