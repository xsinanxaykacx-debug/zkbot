#!/usr/bin/env python3
"""Training-OOF-only multiclass calibration helpers."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

def _normalise(p):
    p=np.clip(np.asarray(p,dtype=float),1e-9,1.0)
    return p/p.sum(axis=1,keepdims=True)

def temperature_fit(p,y):
    p=_normalise(p)
    opt=minimize_scalar(lambda t:log_loss(y,_normalise(np.exp(np.log(p)/t)),labels=[0,1,2]),
      bounds=(0.5,3.0),method="bounded")
    return {"kind":"temperature","temperature":float(opt.x)}

def platt_fit(p,y):
    p=_normalise(p);models=[]
    for k in range(3):
        m=LogisticRegression(C=1.0,solver="lbfgs",max_iter=500)
        m.fit(np.log(p), (np.asarray(y)==k).astype(int));models.append(m)
    return {"kind":"platt","models":models}

def isotonic_fit(p,y):
    p=_normalise(p);models=[]
    for k in range(3):
        m=IsotonicRegression(out_of_bounds="clip",y_min=0.0,y_max=1.0)
        m.fit(p[:,k],(np.asarray(y)==k).astype(float));models.append(m)
    return {"kind":"isotonic","models":models}

def apply_calibrator(p,cal):
    p=_normalise(p);kind=cal["kind"]
    if kind=="temperature":
        z=np.log(p)/float(cal["temperature"]);z-=z.max(axis=1,keepdims=True);return _normalise(np.exp(z))
    if kind=="platt":
        x=np.log(p);q=np.column_stack([m.predict_proba(x)[:,1] for m in cal["models"]]);return _normalise(q)
    if kind=="isotonic":
        q=np.column_stack([m.predict(p[:,i]) for i,m in enumerate(cal["models"])]);return _normalise(q)
    if kind=="none":return p
    raise ValueError(f"Unknown calibrator: {kind}")
