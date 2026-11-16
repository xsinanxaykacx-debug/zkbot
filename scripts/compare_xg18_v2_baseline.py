#!/usr/bin/env python3
"""Paired bootstrap comparison of frozen baseline and XG18 v2 on identical MatchIDs."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import log_loss

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--baseline",type=Path,required=True);ap.add_argument("--candidate",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--bootstrap",type=int,default=20000);a=ap.parse_args()
    b=pd.read_csv(a.baseline);c=pd.read_csv(a.candidate)
    if len(b)!=len(c) or not b.MatchID.astype(str).equals(c.MatchID.astype(str)):raise ValueError("MatchID order mismatch")
    y=np.where(b.actual.astype(str).to_numpy()=="H",0,np.where(b.actual.astype(str).to_numpy()=="D",1,2))
    bp=b[["P_H","P_D","P_A"]].to_numpy();cp=c[["P_H","P_D","P_A"]].to_numpy();bp/=bp.sum(1,keepdims=True);cp/=cp.sum(1,keepdims=True)
    bc=bp.argmax(1)==y;cc=cp.argmax(1)==y;bm=bp.max(1)>=.65;cm=cp.max(1)>=.65;rng=np.random.default_rng(20261116);n=len(y);da=[];dc=[]
    for _ in range(a.bootstrap):
        ix=rng.integers(0,n,n);mb=bm[ix];mc=cm[ix]
        if mb.any() and mc.any():da.append(cc[ix][mc].mean()-bc[ix][mb].mean())
        dc.append(cm[ix].mean()-bm[ix].mean())
    def acc(mask,ok):
        k=int((mask&ok).sum());m=int(mask.sum());return {"n":m,"correct":k,"accuracy":k/m if m else None,"coverage":m/n}
    report={"n":n,"baseline":{"overall_accuracy":float(bc.mean()),"pmax_ge_65":acc(bm,bc),"log_loss":float(log_loss(y,bp,labels=[0,1,2])),"brier":float(np.square(bp-np.eye(3)[y]).sum(1).mean())},
      "candidate":{"overall_accuracy":float(cc.mean()),"pmax_ge_65":acc(cm,cc),"log_loss":float(log_loss(y,cp,labels=[0,1,2])),"brier":float(np.square(cp-np.eye(3)[y]).sum(1).mean())},
      "candidate_minus_baseline":{"pmax_ge_65_accuracy_pp":float((cc[cm].mean()-bc[bm].mean())*100),"bootstrap_95_ci_pp":(np.quantile(da,[.025,.975])*100).tolist(),
        "coverage_pp":float((cm.mean()-bm.mean())*100),"coverage_bootstrap_95_ci_pp":(np.quantile(dc,[.025,.975])*100).tolist()},
      "interpretation":"A positive point estimate is not sufficient; an interval crossing zero does not establish superiority."}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
