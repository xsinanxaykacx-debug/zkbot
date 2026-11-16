#!/usr/bin/env python3
"""Compare odds-aware V3 against the frozen no-odds V3 on identical validation fixtures."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-h),float(c+h)]

def load(path):
    d=pd.read_csv(path)
    required={"MatchID","P_H","P_D","P_A","actual"}
    if required-set(d):raise ValueError(f"{path} missing {sorted(required-set(d))}")
    return d

def metrics(d):
    p=d[["P_H","P_D","P_A"]].to_numpy(float)
    p=np.clip(p,1e-9,1);p/=p.sum(axis=1,keepdims=True)
    y=np.where(d.actual.astype(str).to_numpy()=="H",0,np.where(d.actual.astype(str).to_numpy()=="D",1,2))
    ok=p.argmax(1)==y;conf=p.max(1);mask=conf>=.65
    k=int(ok[mask].sum());n=int(mask.sum())
    return {"n":len(y),"accuracy":float(ok.mean()),"log_loss":float(log_loss(y,p,labels=[0,1,2])),
      "brier":float(np.square(p-np.eye(3)[y]).sum(1).mean()),
      "pmax_ge_65":{"n":n,"coverage":n/len(y),"correct":k,"accuracy":k/n if n else None,"wilson_95":wilson(k,n)}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",type=Path,required=True)
    ap.add_argument("--candidate",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--bootstrap",type=int,default=30000)
    a=ap.parse_args()
    b=load(a.baseline);c=load(a.candidate)
    if not b.MatchID.astype(str).equals(c.MatchID.astype(str)):
        raise ValueError("Baseline and odds-aware candidate MatchIDs/order differ")
    if not b.actual.astype(str).equals(c.actual.astype(str)):
        raise ValueError("Baseline and candidate validation outcomes differ")
    bm=metrics(b);cm=metrics(c)
    y=np.where(b.actual.astype(str).to_numpy()=="H",0,np.where(b.actual.astype(str).to_numpy()=="D",1,2))
    bp=b[["P_H","P_D","P_A"]].to_numpy(float);cp=c[["P_H","P_D","P_A"]].to_numpy(float)
    bp=np.clip(bp,1e-9,1);bp/=bp.sum(1,keepdims=True)
    cp=np.clip(cp,1e-9,1);cp/=cp.sum(1,keepdims=True)
    rng=np.random.default_rng(20261116);delta_acc=[];delta_ll=[]
    for _ in range(a.bootstrap):
        ix=rng.integers(0,len(y),len(y))
        delta_acc.append(float((cp[ix].argmax(1)==y[ix]).mean()-(bp[ix].argmax(1)==y[ix]).mean()))
        delta_ll.append(float(log_loss(y[ix],cp[ix],labels=[0,1,2])-log_loss(y[ix],bp[ix],labels=[0,1,2])))
    result={"validation_rows":len(y),"baseline_no_odds":bm,"candidate_xg_plus_odds":cm,
      "delta_candidate_minus_baseline":{"accuracy_pp":100*(cm["accuracy"]-bm["accuracy"]),
        "accuracy_bootstrap_95_ci_pp":(np.quantile(delta_acc,[.025,.975])*100).tolist(),
        "log_loss":cm["log_loss"]-bm["log_loss"],
        "log_loss_bootstrap_95_ci":np.quantile(delta_ll,[.025,.975]).tolist(),
        "selective_accuracy_pp":100*((cm["pmax_ge_65"]["accuracy"] or 0)-(bm["pmax_ge_65"]["accuracy"] or 0))},
      "decision":"Odds-aware candidate is experimental; do not promote unless paired validation evidence and coverage both support it. Locked final test was not accessed."}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
