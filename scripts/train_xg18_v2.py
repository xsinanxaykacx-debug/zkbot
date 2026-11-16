#!/usr/bin/env python3
"""Train XG18 v2. Tuning and temperature calibration use chronological training folds only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,optuna,numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from xg18_v2_features import build_prematch_features
optuna.logging.set_verbosity(optuna.logging.WARNING)
CATS=["League","HomeTeam","AwayTeam","season"]
FEATURE_COLUMNS=[]

def y_codes(df):
    return np.where(df.FTHG.to_numpy()>df.FTAG.to_numpy(),0,np.where(df.FTHG.to_numpy()==df.FTAG.to_numpy(),1,2))

def date_blocks(df,n_splits=2):
    dates=pd.to_datetime(df.Date,utc=True).dt.floor("D");days=dates.drop_duplicates().sort_values().to_numpy();chunks=np.array_split(days,n_splits+1)
    for i in range(1,n_splits+1):
        td=set(np.concatenate(chunks[:i]));vd=set(chunks[i]);tr=np.flatnonzero(dates.isin(td).to_numpy());va=np.flatnonzero(dates.isin(vd).to_numpy())
        if len(tr)>500 and len(va)>100:yield tr,va

def pipeline(params):
    pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore",min_frequency=2),CATS),
      ("num","passthrough",[c for c in FEATURE_COLUMNS if c not in CATS])])
    clf=XGBClassifier(objective="multi:softprob",num_class=3,eval_metric="mlogloss",tree_method="hist",
      n_estimators=int(params["n_estimators"]),max_depth=int(params["max_depth"]),learning_rate=float(params["learning_rate"]),
      min_child_weight=float(params["min_child_weight"]),subsample=float(params["subsample"]),
      colsample_bytree=float(params["colsample_bytree"]),reg_lambda=float(params["reg_lambda"]),gamma=float(params["gamma"]),
      random_state=20261116,n_jobs=2,verbosity=0)
    return Pipeline([("pre",pre),("model",clf)])

def temp_probs(p,t):
    z=np.log(np.clip(p,1e-12,1.0))/t;z-=z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)

def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-h),float(c+h)]

def metrics(p,y):
    pred=p.argmax(1);conf=p.max(1);ok=pred==y
    out={"n":len(y),"correct":int(ok.sum()),"accuracy":float(ok.mean()),"log_loss":float(log_loss(y,p,labels=[0,1,2])),
      "brier":float(np.square(p-np.eye(3)[y]).sum(1).mean())}
    for t in [.60,.65,.70,.75,.80]:
        m=conf>=t;n=int(m.sum());k=int(ok[m].sum())
        out[f"pmax_ge_{int(t*100)}"]={"n":n,"coverage":n/len(y),"correct":k,"accuracy":k/n if n else None,"wilson_95":wilson(k,n)}
    ece=0.0
    for i in range(10):
        m=(conf>=i/10)&(conf<(i+1)/10 if i<9 else conf<=1.0)
        if m.any():ece+=m.mean()*abs(conf[m].mean()-ok[m].mean())
    out["top_label_ece_10bin"]=float(ece);return out

def main():
    global FEATURE_COLUMNS
    ap=argparse.ArgumentParser();ap.add_argument("--train",type=Path,required=True);ap.add_argument("--validation",type=Path,required=True)
    ap.add_argument("--outdir",type=Path,required=True);ap.add_argument("--trials",type=int,default=4);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(a.train);va=pd.read_csv(a.validation)
    required={"MatchID","League","Season","Date","HomeTeam","AwayTeam","FTHG","FTAG","Home_xG","Away_xG","Home_Shots","Away_Shots","Home_ShotsOnTarget","Away_ShotsOnTarget","Home_BigChance","Away_BigChance"}
    for name,df in [("train",tr),("validation",va)]:
        miss=required-set(df.columns)
        if miss:raise ValueError(f"{name} missing {sorted(miss)}")
    if len(tr)!=12971 or len(va)!=2763:raise ValueError("Unexpected split sizes")
    if pd.to_datetime(tr.Date,utc=True).max()>=pd.to_datetime(va.Date,utc=True).min():raise ValueError("Chronology violation")
    all_df=pd.concat([tr,va],ignore_index=True);feat=build_prematch_features(all_df);FEATURE_COLUMNS=list(feat.columns)
    X=feat.copy()
    for c in CATS:X[c]=X[c].astype(str)
    Xt=X.iloc[:len(tr)].reset_index(drop=True);Xv=X.iloc[len(tr):].reset_index(drop=True);y=y_codes(tr);yv=y_codes(va);folds=list(date_blocks(tr))
    if len(folds)!=2:raise RuntimeError(f"Expected 2 chronological folds, got {len(folds)}")
    def objective(t):
        p={"n_estimators":t.suggest_int("n_estimators",100,200,step=100),"max_depth":t.suggest_int("max_depth",2,4),
          "learning_rate":t.suggest_float("learning_rate",.02,.15,log=True),"min_child_weight":t.suggest_float("min_child_weight",1,12,log=True),
          "subsample":t.suggest_float("subsample",.65,1.0),"colsample_bytree":t.suggest_float("colsample_bytree",.65,1.0),
          "reg_lambda":t.suggest_float("reg_lambda",.5,20,log=True),"gamma":t.suggest_float("gamma",0,.5)}
        losses=[]
        for ix,jx in folds:
            m=pipeline(p);m.fit(Xt.iloc[ix],y[ix]);losses.append(log_loss(y[jx],m.predict_proba(Xt.iloc[jx]),labels=[0,1,2]))
        return float(np.mean(losses))
    study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=20261116));study.optimize(objective,n_trials=a.trials,show_progress_bar=False)
    best=study.best_params;op=[];oy=[]
    for ix,jx in folds:
        m=pipeline(best);m.fit(Xt.iloc[ix],y[ix]);op.append(m.predict_proba(Xt.iloc[jx]));oy.append(y[jx])
    cp=np.vstack(op);cy=np.concatenate(oy);cal=minimize_scalar(lambda t:log_loss(cy,temp_probs(cp,t),labels=[0,1,2]),bounds=(.5,2.5),method="bounded")
    temperature=float(cal.x);final=pipeline(best);final.fit(Xt,y);probs=temp_probs(final.predict_proba(Xv),temperature)
    report={"model":"XG18 v2 XGBoost + pre-match rolling form/Elo/H2H","training_rows":len(tr),"validation_rows":len(va),
      "validation_outcomes_used_for_tuning_or_calibration":False,"feature_policy":"past completed matches only; UTC-day batch before result updates; current-match stats excluded",
      "feature_count":len(FEATURE_COLUMNS),"categorical_features":CATS,"optuna_trials":a.trials,"best_params":best,
      "internal_chronological_cv_logloss":study.best_value,"temperature_from_training_oof_only":temperature,"validation":metrics(probs,yv)}
    joblib.dump(final,a.outdir/"xg18_v2.joblib",compress=3)
    (a.outdir/"xg18_v2_model_metadata.json").write_text(json.dumps({"temperature":temperature,"features":FEATURE_COLUMNS,"categorical_features":CATS},indent=2),encoding="utf-8")
    pred=pd.DataFrame({"MatchID":va.MatchID,"Date":va.Date,"League":va.League,"HomeTeam":va.HomeTeam,"AwayTeam":va.AwayTeam,
      "P_H":probs[:,0],"P_D":probs[:,1],"P_A":probs[:,2],"P_max":probs.max(1),"prediction":np.array(["H","D","A"])[probs.argmax(1)],
      "actual":np.array(["H","D","A"])[yv],"correct":(probs.argmax(1)==yv).astype(int)})
    pred.to_csv(a.outdir/"xg18_v2_validation_predictions.csv",index=False)
    (a.outdir/"xg18_v2_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
