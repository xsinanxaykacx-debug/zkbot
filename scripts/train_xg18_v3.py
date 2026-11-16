#!/usr/bin/env python3
"""XG18 v3: chronological XGB/LGBM/CatBoost ensemble challenger."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,optuna,numpy as np,pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xg18_v3_features import build_prematch_features
from xg18_v3_calibration import temperature_fit,platt_fit,isotonic_fit,apply_calibrator
optuna.logging.set_verbosity(optuna.logging.WARNING)
CATS=["League","HomeTeam","AwayTeam","season"]
FEATURE_COLUMNS=[]
SEED=20261116

def y_codes(df):
    return np.where(df.FTHG.to_numpy()>df.FTAG.to_numpy(),0,np.where(df.FTHG.to_numpy()==df.FTAG.to_numpy(),1,2))

def chronological_folds(df,n_splits=5):
    dates=pd.to_datetime(df.Date,utc=True).dt.floor("D")
    days=np.array(sorted(dates.unique()))
    chunks=np.array_split(days,n_splits+1)
    folds=[]
    for k in range(1,n_splits+1):
        train_days=np.concatenate(chunks[:k]);valid_days=chunks[k]
        tr=np.flatnonzero(dates.isin(train_days).to_numpy())
        va=np.flatnonzero(dates.isin(valid_days).to_numpy())
        if len(tr)<500 or len(va)<100:raise RuntimeError(f"Chronological fold {k} too small: train={len(tr)}, valid={len(va)}")
        if dates.iloc[tr].max()>=dates.iloc[va].min():raise RuntimeError("Fold chronology violation")
        folds.append((tr,va))
    return folds

def make_pipeline(kind,params):
    pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore",min_frequency=2,sparse_output=False),CATS),
      ("num","passthrough",[c for c in FEATURE_COLUMNS if c not in CATS])],sparse_threshold=0.0)
    if kind=="xgb":
        clf=XGBClassifier(objective="multi:softprob",num_class=3,eval_metric="mlogloss",tree_method="hist",
          random_state=SEED,n_jobs=2,verbosity=0,**params)
    elif kind=="lgbm":
        clf=LGBMClassifier(objective="multiclass",num_class=3,verbosity=-1,random_state=SEED,n_jobs=2,
          subsample_freq=1,**params)
    elif kind=="cat":
        clf=CatBoostClassifier(loss_function="MultiClass",verbose=False,random_seed=SEED,thread_count=2,
          allow_writing_files=False,**params)
    else:raise ValueError(kind)
    return Pipeline([("pre",pre),("model",clf)])

def suggest(trial,kind):
    if kind=="xgb":
        return {"n_estimators":trial.suggest_int("n_estimators",180,420,step=120),
          "max_depth":trial.suggest_int("max_depth",2,5),"learning_rate":trial.suggest_float("learning_rate",.02,.12,log=True),
          "min_child_weight":trial.suggest_float("min_child_weight",1,12,log=True),"subsample":trial.suggest_float("subsample",.7,1),
          "colsample_bytree":trial.suggest_float("colsample_bytree",.7,1),"reg_lambda":trial.suggest_float("reg_lambda",.5,20,log=True),
          "gamma":trial.suggest_float("gamma",0,.5)}
    if kind=="lgbm":
        return {"n_estimators":trial.suggest_int("n_estimators",180,420,step=120),
          "num_leaves":trial.suggest_int("num_leaves",7,31),"learning_rate":trial.suggest_float("learning_rate",.02,.12,log=True),
          "min_child_samples":trial.suggest_int("min_child_samples",15,60),"subsample":trial.suggest_float("subsample",.7,1),
          "colsample_bytree":trial.suggest_float("colsample_bytree",.7,1),"reg_lambda":trial.suggest_float("reg_lambda",.5,20,log=True)}
    return {"iterations":trial.suggest_int("iterations",180,420,step=120),"depth":trial.suggest_int("depth",4,7),
      "learning_rate":trial.suggest_float("learning_rate",.02,.12,log=True),
      "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1,15,log=True),"random_strength":trial.suggest_float("random_strength",0,1)}

def fit_oof(kind,params,X,y,folds):
    oof=np.full((len(y),3),np.nan);fold_id=np.full(len(y),-1,dtype=int)
    for k,(tr,va) in enumerate(folds):
        model=make_pipeline(kind,params);model.fit(X.iloc[tr],y[tr]);oof[va]=model.predict_proba(X.iloc[va]);fold_id[va]=k
    return oof,fold_id

def fit_weights(preds,y,mask):
    arr=np.stack([p[mask] for p in preds],axis=0);yy=y[mask]
    def loss(w):return log_loss(yy,np.einsum("m,mnc->nc",w,arr),labels=[0,1,2])
    res=minimize(loss,np.ones(len(preds))/len(preds),method="SLSQP",bounds=[(0,1)]*len(preds),
      constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],options={"maxiter":500,"ftol":1e-10})
    if not res.success:raise RuntimeError("Ensemble weight optimisation failed: "+res.message)
    return (np.clip(res.x,0,1)/np.clip(res.x,0,1).sum()).tolist()

def wilson(k,n):
    if n==0:return [None,None]
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-h),float(c+h)]

def metrics(p,y):
    pred=p.argmax(1);conf=p.max(1);ok=pred==y;out={"n":len(y),"correct":int(ok.sum()),"accuracy":float(ok.mean()),
      "log_loss":float(log_loss(y,p,labels=[0,1,2])),"brier":float(np.square(p-np.eye(3)[y]).sum(1).mean())}
    for t in (.60,.65,.70,.75,.80):
        m=conf>=t;n=int(m.sum());k=int(ok[m].sum())
        out[f"pmax_ge_{int(t*100)}"]={"n":n,"coverage":n/len(y),"correct":k,"accuracy":k/n if n else None,"wilson_95":wilson(k,n)}
    ece=0.0
    for i in range(10):
        m=(conf>=i/10)&(conf<(i+1)/10 if i<9 else conf<=1)
        if m.any():ece+=m.mean()*abs(conf[m].mean()-ok[m].mean())
    out["top_label_ece_10bin"]=float(ece);return out

def main():
    global FEATURE_COLUMNS
    ap=argparse.ArgumentParser();ap.add_argument("--train",type=Path,required=True);ap.add_argument("--validation",type=Path,required=True)
    ap.add_argument("--outdir",type=Path,required=True);ap.add_argument("--trials",type=int,default=3);a=ap.parse_args()
    a.outdir.mkdir(parents=True,exist_ok=True);tr=pd.read_csv(a.train);va=pd.read_csv(a.validation)
    required={"MatchID","League","Season","Date","HomeTeam","AwayTeam","FTHG","FTAG","Home_xG","Away_xG","Home_Shots","Away_Shots","Home_ShotsOnTarget","Away_ShotsOnTarget","Home_BigChance","Away_BigChance"}
    for label,df in (("train",tr),("validation",va)):
        missing=required-set(df.columns)
        if missing:raise ValueError(f"{label} missing {sorted(missing)}")
    if len(tr)<1000 or len(va)<300:raise ValueError(f"Custom chronological split too small: train={len(tr)}, validation={len(va)}")
    if pd.to_datetime(tr.Date,utc=True).max()>=pd.to_datetime(va.Date,utc=True).min():raise ValueError("Train/validation chronology violation")
    all_df=pd.concat([tr,va],ignore_index=True);feat=build_prematch_features(all_df);FEATURE_COLUMNS=list(feat.columns)
    X=feat.copy()
    for c in CATS:X[c]=X[c].astype(str)
    Xt=X.iloc[:len(tr)].reset_index(drop=True);Xv=X.iloc[len(tr):].reset_index(drop=True);y=y_codes(tr);yv=y_codes(va)
    folds=chronological_folds(tr,5);best={};oofs={};fold_ids=None;cv_scores={}
    for kind in ("xgb","lgbm","cat"):
        def objective(t):
            params=suggest(t,kind);losses=[]
            for ix,jx in folds:
                m=make_pipeline(kind,params);m.fit(Xt.iloc[ix],y[ix])
                losses.append(log_loss(y[jx],m.predict_proba(Xt.iloc[jx]),labels=[0,1,2]))
            return float(np.mean(losses))
        study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective,n_trials=a.trials,show_progress_bar=False)
        best[kind]=study.best_params;cv_scores[kind]=study.best_value
        oofs[kind],fold_ids=fit_oof(kind,best[kind],Xt,y,folds)
    valid=fold_ids>=0
    if not valid.any() or set(fold_ids[valid])!={0,1,2,3,4}:raise RuntimeError("Incomplete chronological OOF predictions")
    kinds=("xgb","lgbm","cat");preds=[oofs[k][valid] for k in kinds];yo=y[valid];fo=fold_ids[valid]
    early=fo<=2;late=fo>=3
    early_weights=fit_weights(preds,yo,early)
    early_mix=np.einsum("m,mnc->nc",early_weights,np.stack(preds))
    late_mix=np.einsum("m,mnc->nc",early_weights,np.stack(preds))
    candidates={"none":{"kind":"none"},"temperature":temperature_fit(early_mix[early],yo[early]),
      "platt":platt_fit(early_mix[early],yo[early]),"isotonic":isotonic_fit(early_mix[early],yo[early])}
    cal_scores={name:float(log_loss(yo[late],apply_calibrator(late_mix[late],cal),labels=[0,1,2])) for name,cal in candidates.items()}
    chosen=min(cal_scores,key=cal_scores.get)
    weights=fit_weights(preds,yo,np.ones(len(yo),dtype=bool))
    full_mix=np.einsum("m,mnc->nc",weights,np.stack(preds))
    if chosen=="none":cal={"kind":"none"}
    elif chosen=="temperature":cal=temperature_fit(full_mix,yo)
    elif chosen=="platt":cal=platt_fit(full_mix,yo)
    else:cal=isotonic_fit(full_mix,yo)
    models={}
    for kind in kinds:
        model=make_pipeline(kind,best[kind]);model.fit(Xt,y);models[kind]=model
    individual_probs={k:models[k].predict_proba(Xv) for k in kinds}
    raw=np.einsum("m,mnc->nc",weights,np.stack([individual_probs[k] for k in kinds]))
    probs=apply_calibrator(raw,cal)
    uniform_raw=np.mean(np.stack([individual_probs[k] for k in kinds]),axis=0)
    uniform_probs=apply_calibrator(uniform_raw,cal)
    oof_single_logloss={k:float(log_loss(yo,oofs[k][valid],labels=[0,1,2])) for k in kinds}
    oof_uniform=np.mean(np.stack(preds),axis=0)
    oof_uniform_logloss=float(log_loss(yo,oof_uniform,labels=[0,1,2]))
    oof_weighted_logloss=float(log_loss(yo,full_mix,labels=[0,1,2]))
    report={"model":"XG18 v3 XGBoost+LightGBM+CatBoost ensemble","training_rows":len(tr),"validation_rows":len(va),
      "validation_outcomes_used_for_fit_tuning_weights_or_calibration":False,
      "validation_state_policy":"Past validation-day results update rolling state for later validation dates only; same-day results are hidden until all same-day fixtures are featurized.",
      "feature_policy":"pre-match rolling history only; no current-match statistics; UTC-day batch replay",
      "feature_count":len(FEATURE_COLUMNS),"categorical_features":CATS,"chronological_folds":5,"optuna_trials_per_model":a.trials,
      "best_params":best,"cv_logloss_by_model":cv_scores,
      "oof_single_model_logloss":oof_single_logloss,
      "oof_uniform_ensemble_logloss":oof_uniform_logloss,
      "oof_optimized_ensemble_logloss":oof_weighted_logloss,
      "ensemble_weights":dict(zip(kinds,weights)),
      "validation_metrics_by_single_model":{k:metrics(apply_calibrator(individual_probs[k],cal),yv) for k in kinds},
      "validation_uniform_ensemble":metrics(uniform_probs,yv),
      "calibration_selection":"lowest log loss on final two training OOF folds; candidates none/temperature/Platt/isotonic",
      "calibration_oof_logloss":cal_scores,"selected_calibration":chosen,"validation":metrics(probs,yv)}
    bundle={"models":models,"weights":weights,"calibrator":cal,"features":FEATURE_COLUMNS,"categorical_features":CATS}
    joblib.dump(bundle,a.outdir/"xg18_v3.joblib",compress=3)
    (a.outdir/"xg18_v3_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    pred=pd.DataFrame({"MatchID":va.MatchID,"Date":va.Date,"League":va.League,"HomeTeam":va.HomeTeam,"AwayTeam":va.AwayTeam,
      "P_H":probs[:,0],"P_D":probs[:,1],"P_A":probs[:,2],"P_max":probs.max(1),
      "prediction":np.array(["H","D","A"])[probs.argmax(1)],"actual":np.array(["H","D","A"])[yv],
      "correct":(probs.argmax(1)==yv).astype(int)})
    pred.to_csv(a.outdir/"xg18_v3_validation_predictions.csv",index=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
