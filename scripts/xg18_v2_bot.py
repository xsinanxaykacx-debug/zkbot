#!/usr/bin/env python3
"""Upcoming-fixture CLI for the XG18 v2 model."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from xg18_v2_features import build_prematch_features

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--history",type=Path,required=True);ap.add_argument("--fixtures",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True);ap.add_argument("--metadata",type=Path,default=None)
    ap.add_argument("--threshold",type=float,default=.65);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    hist=pd.read_csv(a.history);fix=pd.read_csv(a.fixtures);rh={"League","Season","Date","HomeTeam","AwayTeam","FTHG","FTAG"};rf={"League","Season","Date","HomeTeam","AwayTeam"}
    if rh-set(hist.columns):raise ValueError(f"History missing {sorted(rh-set(hist.columns))}")
    if rf-set(fix.columns):raise ValueError(f"Fixtures missing {sorted(rf-set(fix.columns))}")
    if fix.empty:raise ValueError("Fixtures are empty")
    hd=pd.to_datetime(hist.Date,utc=True,errors="coerce");fd=pd.to_datetime(fix.Date,utc=True,errors="coerce")
    if hd.isna().any() or fd.isna().any():raise ValueError("Invalid dates")
    if fd.dt.floor("D").nunique()!=1:raise ValueError("Fixtures must contain exactly one UTC date")
    if hd.max().floor("D")>=fd.min().floor("D"):raise ValueError("History must end before fixture date")
    if fix.duplicated(["League","Date","HomeTeam","AwayTeam"]).any():raise ValueError("Duplicate fixtures")
    feats=build_prematch_features(pd.concat([hist,fix],ignore_index=True)).iloc[len(hist):].copy()
    for c in ["League","HomeTeam","AwayTeam","season"]:feats[c]=feats[c].astype(str)
    model=joblib.load(a.model);mp=a.metadata or a.model.with_name("xg18_v2_model_metadata.json")
    meta=json.loads(mp.read_text(encoding="utf-8"));missing=set(meta["features"])-set(feats.columns)
    if missing:raise ValueError(f"Model feature mismatch: {sorted(missing)}")
    feats=feats[meta["features"]];raw=model.predict_proba(feats);t=float(meta["temperature"])
    z=np.log(np.clip(raw,1e-12,1.0))/t;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
    out=fix[["Date","League","HomeTeam","AwayTeam"]].copy();out["P_HOME"]=p[:,0];out["P_DRAW"]=p[:,1];out["P_AWAY"]=p[:,2]
    out["P_MAX"]=p.max(1);out["PREDICTION"]=np.array(["HOME","DRAW","AWAY"])[p.argmax(1)];out["SELECTED"]=p.max(1)>=a.threshold
    a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False);print(out.to_string(index=False))
    print(f"Selected: {int(out.SELECTED.sum())}/{len(out)} at threshold {a.threshold:.0%}")
if __name__=="__main__":main()
