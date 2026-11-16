#!/usr/bin/env python3
"""Carve a forward-in-time confirmation holdout from official training rows only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--train-out",type=Path,required=True)
    ap.add_argument("--confirm-out",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--train-fraction",type=float,default=0.80)
    a=ap.parse_args()
    if not 0.60 <= a.train_fraction <= 0.90:
        raise ValueError("train-fraction must be between 0.60 and 0.90")
    d=pd.read_csv(a.input,low_memory=False)
    if len(d)<2000 or "Date" not in d or "MatchID" not in d:
        raise ValueError("Input must contain at least 2,000 rows, Date and MatchID")
    if d.MatchID.astype(str).duplicated().any():
        raise ValueError("Duplicate MatchID in official training input")
    dates=pd.to_datetime(d.Date,utc=True,errors="coerce").dt.floor("D")
    if dates.isna().any():
        raise ValueError("Invalid dates in official training input")
    counts=dates.value_counts().sort_index()
    cumulative=counts.cumsum()
    cut_pos=int((cumulative < len(d)*a.train_fraction).sum())
    cut_pos=max(1,min(cut_pos,len(counts)-1))
    cut_date=counts.index[cut_pos]
    train=d.loc[dates<cut_date].copy()
    confirm=d.loc[dates>=cut_date].copy()
    if len(train)<1000 or len(confirm)<300:
        raise ValueError(f"Unsafe split sizes: train={len(train)}, confirmation={len(confirm)}")
    if pd.to_datetime(train.Date,utc=True).max()>=pd.to_datetime(confirm.Date,utc=True).min():
        raise RuntimeError("Chronology boundary violation")
    if set(train.MatchID.astype(str)) & set(confirm.MatchID.astype(str)):
        raise RuntimeError("MatchID overlap")
    a.train_out.parent.mkdir(parents=True,exist_ok=True)
    a.confirm_out.parent.mkdir(parents=True,exist_ok=True)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    train.to_csv(a.train_out,index=False)
    confirm.to_csv(a.confirm_out,index=False)
    report={"source":"official XG18 training partition only","final_test_rows_used":0,
      "validation_partition_used":False,"train_rows":len(train),"confirmation_rows":len(confirm),
      "train_start":str(pd.to_datetime(train.Date,utc=True).min()),
      "train_end":str(pd.to_datetime(train.Date,utc=True).max()),
      "confirmation_start":str(pd.to_datetime(confirm.Date,utc=True).min()),
      "confirmation_end":str(pd.to_datetime(confirm.Date,utc=True).max()),
      "cut_date_utc":str(cut_date),"train_fraction_actual":len(train)/len(d),
      "confirmation_fraction_actual":len(confirm)/len(d),"same_day_split":False}
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
