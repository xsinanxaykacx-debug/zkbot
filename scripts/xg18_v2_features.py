#!/usr/bin/env python3
"""Leakage-safe pre-match feature generation. Same UTC-day fixtures are featurized before results update state."""
from __future__ import annotations
from collections import defaultdict, deque
import numpy as np
import pandas as pd

METRICS=["gf","ga","xgf","xga","shotsf","shotsa","sotf","sota","bigf","biga","points"]
DEFAULTS={"gf":1.35,"ga":1.35,"xgf":1.35,"xga":1.35,"shotsf":12.0,"shotsa":12.0,"sotf":4.0,"sota":4.0,"bigf":1.2,"biga":1.2,"points":1.25}

def _num(row,key,default=np.nan):
    v=row.get(key,default)
    try:return float(v) if pd.notna(v) else default
    except (TypeError,ValueError):return default

def _team_observation(row,side):
    other="Away" if side=="Home" else "Home"
    gf=_num(row,f"{side}Goals",_num(row,"FTHG" if side=="Home" else "FTAG"))
    ga=_num(row,f"{other}Goals",_num(row,"FTAG" if side=="Home" else "FTHG"))
    pts=3.0 if gf>ga else (1.0 if gf==ga else 0.0)
    return dict(gf=gf,ga=ga,xgf=_num(row,f"{side}_xG"),xga=_num(row,f"{other}_xG"),
      shotsf=_num(row,f"{side}_Shots"),shotsa=_num(row,f"{other}_Shots"),
      sotf=_num(row,f"{side}_ShotsOnTarget"),sota=_num(row,f"{other}_ShotsOnTarget"),
      bigf=_num(row,f"{side}_BigChance"),biga=_num(row,f"{other}_BigChance"),points=pts)

def _mean(items,metric,default):
    vals=[x[metric] for x in items if np.isfinite(x[metric])]
    return float(np.mean(vals)) if vals else float(default)

def build_prematch_features(matches:pd.DataFrame)->pd.DataFrame:
    required={"League","Date","HomeTeam","AwayTeam"}
    missing=required-set(matches.columns)
    if missing:raise ValueError(f"Missing columns: {sorted(missing)}")
    df=matches.copy();df["_date"]=pd.to_datetime(df.Date,utc=True,errors="coerce")
    if df._date.isna().any():raise ValueError("Invalid match dates")
    df["_day"]=df._date.dt.floor("D");df["_row_order"]=np.arange(len(df))
    df=df.sort_values(["_day","_date","_row_order"],kind="stable")
    team=defaultdict(lambda:deque(maxlen=10))
    venue=defaultdict(lambda:{"H":deque(maxlen=5),"A":deque(maxlen=5)})
    last={};elo=defaultdict(lambda:1500.0);h2h=defaultdict(lambda:deque(maxlen=5));league_hist=defaultdict(lambda:deque(maxlen=100))
    records={}
    def team_features(lg,name,side,day,prefix):
        hist=list(team[(lg,name)]);vh=list(venue[(lg,name)][side]);out={}
        for n in (3,5,10):
            sub=hist[-n:]
            for m in METRICS:out[f"{prefix}_{m}_last{n}"]=_mean(sub,m,DEFAULTS[m])
        for m in METRICS:out[f"{prefix}_{m}_venue5"]=_mean(vh,m,_mean(hist,m,DEFAULTS[m]))
        out[f"{prefix}_matches_prior"]=len(hist)
        lp=last.get((lg,name));out[f"{prefix}_rest_days"]=min(60.0,max(0.0,(day-lp).total_seconds()/86400)) if lp is not None else 7.0
        out[f"{prefix}_elo"]=elo[(lg,name)]
        return out
    for day,group in df.groupby("_day",sort=True):
        day_rows=[]
        for _,r in group.iterrows():
            lg=str(r.League);h=str(r.HomeTeam);a=str(r.AwayTeam);f={"League":lg,"HomeTeam":h,"AwayTeam":a}
            f.update(team_features(lg,h,"H",day,"home"));f.update(team_features(lg,a,"A",day,"away"))
            f["elo_diff"]=elo[(lg,h)]+65.0-elo[(lg,a)]
            pair=tuple(sorted((h,a)));prior=list(h2h[(lg,pair)])
            oriented=[x if h==pair[0] else (-x[0],3.0-x[1] if x[1] in (0.0,3.0) else 1.0) for x in prior]
            f["h2h_n"]=len(prior);f["h2h_goal_diff_last5"]=float(np.mean([x[0] for x in oriented])) if oriented else 0.0
            f["h2h_points_home_last5"]=float(np.mean([x[1] for x in oriented])) if oriented else 1.0
            lh=list(league_hist[lg]);f["league_avg_total_goals_prior"]=float(np.mean([x[0] for x in lh])) if lh else 2.65
            f["league_avg_total_xg_prior"]=float(np.mean([x[1] for x in lh])) if lh else 2.65
            f["month"]=int(r._date.month);f["weekday"]=int(r._date.dayofweek);f["season"]=str(r.get("Season","unknown"))
            records[int(r._row_order)]=f;day_rows.append(r)
        # No result from this UTC date is applied until every fixture on the date has features.
        for r in day_rows:
            if pd.isna(r.get("FTHG",np.nan)) or pd.isna(r.get("FTAG",np.nan)):continue
            lg=str(r.League);h=str(r.HomeTeam);a=str(r.AwayTeam);hg=float(r.FTHG);ag=float(r.FTAG)
            ho=_team_observation(r,"Home");ao=_team_observation(r,"Away")
            team[(lg,h)].append(ho);venue[(lg,h)]["H"].append(ho)
            team[(lg,a)].append(ao);venue[(lg,a)]["A"].append(ao)
            last[(lg,h)]=day;last[(lg,a)]=day
            rh=elo[(lg,h)];ra=elo[(lg,a)];eh=1/(1+10**(-(rh+65-ra)/400))
            score=1.0 if hg>ag else (0.5 if hg==ag else 0.0);k=20*np.log1p(max(1.0,abs(hg-ag)))
            delta=k*(score-eh);elo[(lg,h)]=rh+delta;elo[(lg,a)]=ra-delta
            pair=tuple(sorted((h,a)));gd=hg-ag if h==pair[0] else ag-hg
            pts=(3.0 if hg>ag else (1.0 if hg==ag else 0.0)) if h==pair[0] else (3.0 if ag>hg else (1.0 if ag==hg else 0.0))
            h2h[(lg,pair)].append((gd,pts))
            league_hist[lg].append((hg+ag,_num(r,"Home_xG",1.3)+_num(r,"Away_xG",1.3)))
    return pd.DataFrame.from_dict(records,orient="index").sort_index()
