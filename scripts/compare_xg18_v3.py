#!/usr/bin/env python3
"""Paired fixture bootstrap for frozen baseline, V2 and V3 selective performance."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import log_loss

def load(path):
    d=pd.read_csv(path)
    if "actual" not in d:raise ValueError(f"{path} has no actual column")
    if "P_max" not in d:d["P_max"]=d[["P_H","P_D","P_A"]].max(axis=1)
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--baseline",type=Path,required=True);ap.add_argument("--v2",type=Path,required=True)
    ap.add_argument("--v3",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--bootstrap",type=int,default=30000);a=ap.parse_args()
    frames=[load(x) for x in (a.baseline,a.v2,a.v3)]
    ids=frames[0].MatchID.astype(str)
    for d in frames[1:]:
        if not ids.equals(d.MatchID.astype(str)):raise ValueError("MatchID order mismatch; comparisons must use identical fixtures")
    y=np.where(frames[0].actual.astype(str).to_numpy()=="H",0,np.where(frames[0].actual.astype(str).to_numpy()=="D",1,2))
    probs=[d[["P_H","P_D","P_A"]].to_numpy(dtype=float) for d in frames]
    for p in probs:p/=p.sum(1,keepdims=True)
    masks=[p.max(1)>=.65 for p in probs];oks=[p.argmax(1)==y for p in probs];n=len(y)
    def summarize(p,m,ok):
        k=int((m&ok).sum());count=int(m.sum())
        return {"n":count,"correct":k,"accuracy":k/count if count else None,"coverage":count/n,
          "wilson_95":wilson(k,count),"overall_accuracy":float(ok.mean()),
          "log_loss":float(log_loss(y,p,labels=[0,1,2])),"brier":float(np.square(p-np.eye(3)[y]).sum(1).mean())}
    def wilson(k,n0):
        if not n0:return [None,None]
        z=1.959963984540054;p0=k/n0;d0=1+z*z/n0;c=(p0+z*z/(2*n0))/d0;h=z*np.sqrt(p0*(1-p0)/n0+z*z/(4*n0*n0))/d0
        return [float(c-h),float(c+h)]
    rng=np.random.default_rng(20261116);pairs={}
    for i,j in ((0,1),(0,2),(1,2)):
        da=[];dc=[]
        for _ in range(a.bootstrap):
            ix=rng.integers(0,n,n);mi=masks[i][ix];mj=masks[j][ix]
            if mi.any() and mj.any():da.append(oks[j][ix][mj].mean()-oks[i][ix][mi].mean())
            dc.append(masks[j][ix].mean()-masks[i][ix].mean())
        pairs[f"{('baseline','v2','v3')[i]}_to_{('baseline','v2','v3')[j]}"]={
          "selective_accuracy_delta_pp":float((oks[j][masks[j]].mean()-oks[i][masks[i]].mean())*100),
          "paired_bootstrap_95_ci_pp":(np.quantile(da,[.025,.975])*100).tolist(),
          "coverage_delta_pp":float((masks[j].mean()-masks[i].mean())*100),
          "coverage_bootstrap_95_ci_pp":(np.quantile(dc,[.025,.975])*100).tolist()}
    report={"n_identical_fixtures":n,"threshold":.65,
      "models":{name:summarize(p,m,o) for name,p,m,o in zip(("baseline","v2","v3"),probs,masks,oks)},
      "paired_bootstrap":pairs,
      "interpretation":"Selective accuracy is conditional on each model's own threshold-selected subset. A CI excluding zero supports a difference, not a guarantee of future performance. Coverage must be reviewed separately."}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
