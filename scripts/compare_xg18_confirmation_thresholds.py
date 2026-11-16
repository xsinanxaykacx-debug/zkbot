#!/usr/bin/env python3
"""Pre-registered paired confirmation of fixed Pmax thresholds 0.64 vs 0.65."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

THRESHOLDS=(0.64,0.65)
def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-h),float(c+h)]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--bootstrap",type=int,default=30000)
    a=ap.parse_args()
    d=pd.read_csv(a.predictions)
    required={"MatchID","P_max","prediction","actual"}
    if required-set(d):raise ValueError(f"Missing columns: {sorted(required-set(d))}")
    if d.MatchID.astype(str).duplicated().any():raise ValueError("Duplicate confirmation MatchID")
    conf=pd.to_numeric(d.P_max,errors="coerce").to_numpy(float)
    if not np.isfinite(conf).all() or ((conf<0)|(conf>1)).any():raise ValueError("Invalid P_max")
    ok=d.prediction.astype(str).to_numpy()==d.actual.astype(str).to_numpy()
    n=len(d)
    if n<300:raise ValueError("Confirmation set too small")
    masks={t:conf>=t for t in THRESHOLDS}
    metrics={}
    for t,m in masks.items():
        k=int(ok[m].sum());cnt=int(m.sum())
        metrics[f"{t:.2f}"]={"selected":cnt,"coverage":cnt/n,"correct":k,
          "accuracy":k/cnt if cnt else None,"wilson_95":wilson(k,cnt)}
    rng=np.random.default_rng(20261117)
    delta_acc=np.empty(a.bootstrap);delta_cov=np.empty(a.bootstrap)
    for b in range(a.bootstrap):
        ix=rng.integers(0,n,n)
        m64=masks[0.64][ix];m65=masks[0.65][ix];o=ok[ix]
        a64=o[m64].mean() if m64.any() else np.nan
        a65=o[m65].mean() if m65.any() else np.nan
        delta_acc[b]=a64-a65
        delta_cov[b]=m64.mean()-m65.mean()
    valid=np.isfinite(delta_acc)
    da=delta_acc[valid]
    p=float(min(1.0,2*min((np.sum(da<=0)+1)/(len(da)+1),(np.sum(da>=0)+1)/(len(da)+1))))
    result={"n":n,"thresholds_fixed_before_confirmation":[0.64,0.65],
      "threshold_64_minus_65":{"accuracy_difference_pp":100*(metrics["0.64"]["accuracy"]-metrics["0.65"]["accuracy"]),
        "coverage_difference_pp":100*(metrics["0.64"]["coverage"]-metrics["0.65"]["coverage"]),
        "accuracy_bootstrap_95_ci_pp":(np.quantile(da,[.025,.975])*100).tolist(),
        "coverage_bootstrap_95_ci_pp":(np.quantile(delta_cov,[.025,.975])*100).tolist(),
        "two_sided_bootstrap_p":p},
      "metrics":metrics,"bootstrap_replicates":a.bootstrap,
      "note":"This is a forward-in-time holdout carved only from the official training partition. No official validation or final-test rows were used to fit this confirmation model."}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
