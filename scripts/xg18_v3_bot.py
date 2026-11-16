#!/usr/bin/env python3
"""Upcoming fixture CLI for XG18 v3 ensemble."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib,numpy as np,pandas as pd
from xg18_v3_features import build_prematch_features
from xg18_v3_calibration import apply_calibrator

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--history",type=Path,required=True);ap.add_argument("--fixtures",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True);ap.add_argument("--threshold",type=float,default=.65);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    hist=pd.read_csv(a.history);fix=pd.read_csv(a.fixtures)
    rh={"League","Season","Date","HomeTeam","AwayTeam","FTHG","FTAG"};rf={"League","Season","Date","HomeTeam","AwayTeam"}
    if rh-set(hist.columns):raise ValueError(f"History missing {sorted(rh-set(hist.columns))}")
    if rf-set(fix.columns):raise ValueError(f"Fixtures missing {sorted(rf-set(fix.columns))}")
    if fix.empty:raise ValueError("Fixtures are empty")
    hd=pd.to_datetime(hist.Date,utc=True,errors="coerce");fd=pd.to_datetime(fix.Date,utc=True,errors="coerce")
    if hd.isna().any() or fd.isna().any():raise ValueError("Invalid match dates")
    if fd.dt.floor("D").nunique()!=1:raise ValueError("Run one UTC fixture date at a time")
    if hd.max().floor("D")>=fd.min().floor("D"):raise ValueError("History must end before fixture date")
    if fix.duplicated(["League","Date","HomeTeam","AwayTeam"]).any():raise ValueError("Duplicate fixtures")
    feats=build_prematch_features(pd.concat([hist,fix],ignore_index=True)).iloc[len(hist):].copy()
    for c in ("League","HomeTeam","AwayTeam","season"):feats[c]=feats[c].astype(str)
    bundle=joblib.load(a.model);missing=set(bundle["features"])-set(feats.columns)
    if missing:raise ValueError(f"Feature mismatch: {sorted(missing)}")
    X=feats[bundle["features"]];kinds=("xgb","lgbm","cat")
    raw=np.einsum("m,mnc->nc",bundle["weights"],np.stack([bundle["models"][k].predict_proba(X) for k in kinds]))
    p=apply_calibrator(raw,bundle["calibrator"])
    out=fix[["Date","League","HomeTeam","AwayTeam"]].copy();out["P_HOME"]=p[:,0];out["P_DRAW"]=p[:,1];out["P_AWAY"]=p[:,2]
    out["P_MAX"]=p.max(1);out["PREDICTION"]=np.array(["HOME","DRAW","AWAY"])[p.argmax(1)];out["SELECTED"]=p.max(1)>=a.threshold
    a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False);print(out.to_string(index=False))
    print(f"Selected: {int(out.SELECTED.sum())}/{len(out)} at threshold {a.threshold:.0%}")
if __name__=="__main__":main()
