#!/usr/bin/env python3
"""XG18 frozen shadow runner. No betting, tuning, or sealed-test access."""
import argparse, importlib.util, json, math, sys, uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

VERSION="zk2-xg18-frozen-ensemble-v1"
FROZEN_COMMIT="6053f2878cff4c0d73bc16ce65b2bb71ba26f742"
WEIGHTS={"team_poisson":0.13958888173927925,"dixon_coles":0.0,"elo":0.8604111182607207}
K,HA,NU=32.0,80.0,0.8

def utc(v):
    t=pd.to_datetime(v,utc=True,errors="raise")
    if pd.isna(t): raise ValueError(f"Invalid UTC timestamp: {v}")
    return t

def read_events(p):
    p=Path(p)
    if not p.exists(): return []
    out=[]
    for n,line in enumerate(p.read_text(encoding="utf-8").splitlines(),1):
        if line.strip():
            try: out.append(json.loads(line))
            except Exception as e: raise ValueError(f"Corrupt ledger line {n}: {e}")
    return out

def append(p,e):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    old=read_events(p); mid=e.get("match_id")
    if any(x.get("event_id")==e["event_id"] for x in old): raise ValueError("Duplicate event_id")
    typ=e["event_type"]
    if typ in ("prediction","result") and any(x["event_type"]==typ and x.get("match_id")==mid for x in old):
        raise ValueError(f"Immutable {typ} already exists for {mid}")
    if typ=="odds_snapshot" and e["is_closing"] and any(x["event_type"]==typ and x.get("match_id")==mid and x.get("is_closing") for x in old):
        raise ValueError(f"Closing odds already exists for {mid}")
    with p.open("a",encoding="utf-8") as f: f.write(json.dumps(e,ensure_ascii=False,allow_nan=False)+"\n")

def model_module(p):
    p=Path(p)
    if not p.is_file(): raise FileNotFoundError(f"{p} missing; assemble/extract research/xg18-frozen-source.zip first")
    s=importlib.util.spec_from_file_location("xg18_frozen_model",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def elo_p(rh,ra):
    z=math.log(10)/400*(rh+HA-ra); eh=math.exp(max(-30,min(30,z/2))); ea=math.exp(max(-30,min(30,-z/2)))
    return np.array([eh,NU,ea],float)/(eh+NU+ea)

def elo_replay(hist,future):
    ratings=defaultdict(lambda:1500.0)
    for _,g in hist.sort_values("DateParsed",kind="mergesort").groupby(hist["DateParsed"].dt.floor("D"),sort=True):
        delta=defaultdict(float)
        for r in g.itertuples():
            h=(r.LeagueKey,r.HomeKey); a=(r.LeagueKey,r.AwayKey); p=elo_p(ratings[h],ratings[a])
            actual=1.0 if r.FTHG>r.FTAG else (0.5 if r.FTHG==r.FTAG else 0.0)
            d=K*(actual-(p[0]+0.5*p[1])); delta[h]+=d; delta[a]-=d
        for key,d in delta.items(): ratings[key]+=d
    out={}
    for _,g in future.sort_values("DateParsed",kind="mergesort").groupby(future["DateParsed"].dt.floor("D"),sort=True):
        for r in g.itertuples(): out[str(r.MatchID)]=elo_p(ratings[(r.LeagueKey,r.HomeKey)],ratings[(r.LeagueKey,r.AwayKey)]).tolist()
    return out

def predict(a):
    m=model_module(a.model_module); raw=pd.read_csv(a.history); fx=pd.read_csv(a.fixtures)
    hc={"League","Date","HomeTeam","AwayTeam","FTHG","FTAG"}; fc={"MatchID","League","Date","HomeTeam","AwayTeam"}
    if not hc.issubset(raw) or not fc.issubset(fx): raise ValueError("History/fixture columns do not match README schema")
    if fx.MatchID.isna().any() or fx.MatchID.astype(str).duplicated().any(): raise ValueError("MatchID must be non-empty and unique")
    hist=m.prep(raw); fut=m.prep(fx); now=utc(a.generated_at or datetime.now(timezone.utc).isoformat())
    if hist.DateParsed.isna().any() or fut.DateParsed.isna().any() or hist[["FTHG","FTAG"]].isna().any().any(): raise ValueError("Invalid dates or missing historical scores")
    if len(hist)!=15734 or hist.DateParsed.min().date().isoformat()!="2022-07-15" or hist.DateParsed.max().date().isoformat()!="2026-03-14":
        raise ValueError("Fit history must be the fixed 15,734 train+validation rows dated 2022-07-15..2026-03-14; never append shadow results to this file")
    if hist.DateParsed.max()>=fut.DateParsed.min() or (fut.DateParsed<=now).any(): raise ValueError("Chronology guard failed: fixtures must be future and after fixed history")
    if fut[["LeagueKey","HomeKey","AwayKey"]].isna().any().any(): raise ValueError("Fixture identity is incomplete")
    ledger=read_events(a.ledger); oldpred={x["match_id"] for x in ledger if x["event_type"]=="prediction"}
    tm=m.TeamPoisson(alpha=10).fit(hist); lh,la=tm.predict(fut); joint=m.score_joint(lh,la,rho=0.0,maxg=12)
    # Frozen manifest is 1X2 only. OU2.5/BTTS are auxiliary Team-Poisson diagnostics.
    dummy=m.metrics_joint(np.zeros(len(fut),dtype=int),np.zeros(len(fut),dtype=int),joint)["probs_1x2"]
    min_day=fut.DateParsed.dt.floor("D").min(); results={x["match_id"]:x for x in ledger if x["event_type"]=="result"}
    predmap={x["match_id"]:x for x in ledger if x["event_type"]=="prediction"}; online=[]
    for mid,res in results.items():
        pr=predmap.get(mid)
        if not pr: continue
        if utc(pr["kickoff_utc"]).floor("D")>=min_day or utc(res["result_available_at"])>now: continue
        online.append({"League":pr["league"],"Date":pr["kickoff_utc"],"HomeTeam":pr["home_team"],"AwayTeam":pr["away_team"],"FTHG":res["home_goals"],"FTAG":res["away_goals"]})
    eh=m.prep(pd.concat([raw,pd.DataFrame(online)],ignore_index=True)) if online else hist
    ep=elo_replay(eh,fut); events=[]
    for i,r in enumerate(fut.itertuples()):
        mid=str(r.MatchID)
        if mid in oldpred: raise ValueError(f"Prediction already exists for {mid}; no overwrite")
        elo=np.array(ep[mid]); team=dummy[i]; ens=WEIGHTS["team_poisson"]*team+WEIGHTS["elo"]*elo; ens/=ens.sum()
        g=np.arange(joint.shape[1]); totals=np.add.outer(g,g); p_over=float((joint[i]*(totals>2)).sum())
        p_btts=float(1-joint[i,0,:].sum()-joint[i,:,0].sum()+joint[i,0,0])
        events.append({"event_id":str(uuid.uuid4()),"event_type":"prediction","generated_at":now.isoformat().replace("+00:00","Z"),
          "model_version":VERSION,"frozen_model_commit":FROZEN_COMMIT,"match_id":mid,"league":str(r.League),
          "kickoff_utc":r.DateParsed.isoformat().replace("+00:00","Z"),"home_team":str(r.HomeTeam),"away_team":str(r.AwayTeam),
          "probabilities_1x2":dict(zip(("home","draw","away"),map(float,ens))),
          "elo_reference_1x2":dict(zip(("home","draw","away"),map(float,elo))),
          "team_poisson_component_1x2":dict(zip(("home","draw","away"),map(float,team))),
          "auxiliary_team_poisson_only":{"over_2_5":p_over,"btts_yes":p_btts,"not_part_of_frozen_ensemble":True},
          "weights":WEIGHTS,"hyperparameters":{"team_poisson_alpha":10,"elo_k":K,"elo_home_advantage":HA,"elo_draw_nu":NU}})
    for e in events: append(a.ledger,e)
    print(json.dumps({"status":"ok","predictions_written":len(events),"model_version":VERSION,"betting_enabled":False},indent=2))

def identity_key(value):
    import unicodedata
    value=unicodedata.normalize("NFKD",str(value)).encode("ascii","ignore").decode("ascii").casefold()
    return "".join(ch for ch in value if ch.isalnum())

def add_odds(a):
    rows=pd.read_csv(a.input,keep_default_na=False)
    need={"MatchID","captured_at","home_odds","draw_odds","away_odds","source"}
    if not need.issubset(rows): raise ValueError(f"Odds CSV missing {sorted(need-set(rows))}")
    ev=read_events(a.ledger); preds={x["match_id"]:x for x in ev if x["event_type"]=="prediction"}
    provider_default=a.provider_id or "manual_csv"
    for r in rows.to_dict("records"):
        mid=str(r["MatchID"]).strip()
        if mid not in preds: raise ValueError(f"No prediction for {mid}")
        source=str(r["source"]).strip().casefold()
        if source not in ("pinnacle","third_party"): raise ValueError("source must be pinnacle or third_party")
        provider_id=str(r.get("provider_id") or provider_default).strip()
        if not provider_id: raise ValueError("provider_id is required")
        if source=="pinnacle" and provider_id!="pinnacle_official": raise ValueError("Only provider_id=pinnacle_official may be labelled as Pinnacle-origin data")
        pinnacle_id=str(r.get("pinnacle_source_id") or "").strip() or None
        t=utc(r["captured_at"]); kickoff=utc(r.get("kickoff_utc") or preds[mid]["kickoff_utc"])
        if kickoff!=utc(preds[mid]["kickoff_utc"]): raise ValueError("CSV kickoff_utc does not match the immutable fixture prediction")
        for field, expected in (("provider_league",preds[mid]["league"]),("provider_home_team",preds[mid]["home_team"]),("provider_away_team",preds[mid]["away_team"])):
            actual=str(r.get(field) or "").strip()
            if actual and identity_key(actual)!=identity_key(expected): raise ValueError(f"Provider fixture mismatch for {mid}: {field}={actual!r}, expected={expected!r}; manual review required")
        if t>=kickoff: raise ValueError("Odds timestamp must precede kickoff")
        o=np.array([float(r[k]) for k in ("home_odds","draw_odds","away_odds")])
        if not np.isfinite(o).all() or (o<=1).any(): raise ValueError("Decimal odds must be finite and >1")
        inv=1/o; fair=inv/inv.sum()
        close=str(r.get("is_closing","false")).casefold() in ("true","1","yes","y")
        if close and source!="pinnacle": raise ValueError("Third-party closing claims cannot be labelled as Pinnacle closing")
        captured= t.isoformat().replace("+00:00","Z")
        append(a.ledger,{"event_id":str(uuid.uuid4()),"event_type":"odds_snapshot",
          "recorded_at":datetime.now(timezone.utc).isoformat(),"match_id":mid,"source":source,
          "provider_id":provider_id,"pinnacle_source_id":pinnacle_id,"captured_at":captured,
          "kickoff_utc":kickoff.isoformat().replace("+00:00","Z"),"is_closing":close,
          "observed_closing_proxy":False,"snapshot_kind":str(r.get("snapshot_kind") or "live_snapshot"),
          "provider_league":str(r.get("provider_league") or ""),"provider_home_team":str(r.get("provider_home_team") or ""),
          "provider_away_team":str(r.get("provider_away_team") or ""),"timestamp_semantics":str(r.get("timestamp_semantics") or "legacy_csv_unverified"),
          "decimal_odds":o.tolist(),"fair_probs_1x2":fair.tolist(),
          "method":"proportional_overround_removal"})
    print(json.dumps({"status":"ok","odds_snapshots_written":len(rows),
      "provider_id":provider_default,"closing_proxy":"computed only after kickoff from the latest eligible snapshot within 120 seconds; never official Pinnacle closing"},indent=2))

def settle(a):
    rows=pd.read_csv(a.input); need={"MatchID","FTHG","FTAG","result_available_at"}
    if not need.issubset(rows): raise ValueError(f"Results CSV missing {sorted(need-set(rows))}")
    ev=read_events(a.ledger); preds={x["match_id"]:x for x in ev if x["event_type"]=="prediction"}
    for r in rows.to_dict("records"):
        mid=str(r["MatchID"])
        if mid not in preds or pd.isna(r["FTHG"]) or pd.isna(r["FTAG"]): raise ValueError(f"Missing prediction or score for {mid}")
        t=utc(r["result_available_at"])
        if t<utc(preds[mid]["kickoff_utc"]): raise ValueError("Result availability cannot precede kickoff")
        append(a.ledger,{"event_id":str(uuid.uuid4()),"event_type":"result","recorded_at":datetime.now(timezone.utc).isoformat(),
          "match_id":mid,"result_available_at":t.isoformat().replace("+00:00","Z"),"home_goals":int(r["FTHG"]),"away_goals":int(r["FTAG"])})
    print(json.dumps({"status":"ok","results_written":len(rows)},indent=2))

def metric(p,y):
    one=np.eye(3)[y]; conf=p.max(1); cor=(p.argmax(1)==y); e=0.0
    for i in range(10):
        mask=(conf>=i/10)&((conf<(i+1)/10) if i<9 else (conf<=1))
        if mask.any(): e+=mask.mean()*abs(conf[mask].mean()-cor[mask].mean())
    return {"n":len(y),"log_loss":float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()),
      "brier":float(((p-one)**2).sum(1).mean()),"accuracy":float(cor.mean()),"ece_10bin":float(e)}

def report(a):
    ev=read_events(a.ledger); preds={x["match_id"]:x for x in ev if x["event_type"]=="prediction"}
    res={x["match_id"]:x for x in ev if x["event_type"]=="result"}; odds={x["match_id"]:x for x in ev if x["event_type"]=="odds_snapshot" and x.get("is_closing") and x.get("source")=="pinnacle" and x.get("provider_id")=="pinnacle_official"}
    ids=sorted(set(preds)&set(res))
    if not ids: print(json.dumps({"status":"insufficient_data","settled":0},indent=2)); return
    def yy(ms): return np.array([0 if res[x]["home_goals"]>res[x]["away_goals"] else (1 if res[x]["home_goals"]==res[x]["away_goals"] else 2) for x in ms])
    def pp(ms,key): return np.array([[preds[x][key][k] for k in ("home","draw","away")] for x in ms])
    out={"model_version":VERSION,"settled_matches":len(ids),"ensemble_1x2":metric(pp(ids,"probabilities_1x2"),yy(ids)),
      "elo_reference_1x2":metric(pp(ids,"elo_reference_1x2"),yy(ids)),"market_comparison":{"n":0,"status":"no_verified_closing_odds"},"by_league":{}}
    # These are Team-Poisson-only auxiliary markets, not frozen ensemble outputs.
    goals=np.array([res[x]["home_goals"]+res[x]["away_goals"] for x in ids])
    def binary_metric(prob,target):
        prob=np.clip(np.asarray(prob,dtype=float),1e-15,1-1e-15); target=np.asarray(target,dtype=int)
        return {"n":len(target),"log_loss":float(-(target*np.log(prob)+(1-target)*np.log(1-prob)).mean()),
                "brier":float(((prob-target)**2).mean())}
    aux=[preds[x]["auxiliary_team_poisson_only"] for x in ids]
    out["auxiliary_team_poisson_only"]={"ou25":binary_metric([x["over_2_5"] for x in aux],goals>2),
      "btts_yes":binary_metric([x["btts_yes"] for x in aux],[(res[x]["home_goals"]>0 and res[x]["away_goals"]>0) for x in ids]),
      "note":"Not part of the frozen 1X2 ensemble; no Pinnacle OU/BTTS comparison in this release."}
    mi=[x for x in ids if x in odds]
    if mi: out["market_comparison"]={"n":len(mi),"source":"Pinnacle","de_vig":"proportional_overround_removal",
      "ensemble_1x2":metric(pp(mi,"probabilities_1x2"),yy(mi)),"pinnacle_fair_1x2":metric(np.array([odds[x]["fair_probs_1x2"] for x in mi]),yy(mi))}
    # Observed closing proxy is a separate diagnostic, never official Pinnacle closing.
    proxy={}
    for mid in ids:
        kickoff=utc(preds[mid]["kickoff_utc"])
        eligible=[]
        for snap in ev:
            if snap.get("event_type")!="odds_snapshot" or snap.get("match_id")!=mid: continue
            if snap.get("source")=="pinnacle" and snap.get("provider_id")=="pinnacle_official" and snap.get("is_closing"): continue
            captured=utc(snap["captured_at"]); age=(kickoff-captured).total_seconds()
            if 0 <= age <= 120: eligible.append((captured,snap))
        if eligible:
            snap=max(eligible,key=lambda z:z[0])[1]
            proxy[mid]=snap
    out["observed_closing_proxy"]={"window_seconds":120,"n":len(proxy),
      "status":"proxy_only_not_official_pinnacle_closing",
      "providers":dict((pid,sum(1 for x in proxy.values() if x.get("provider_id")==pid))
        for pid in sorted({x.get("provider_id","unknown") for x in proxy.values()}))}
    if proxy:
        px=sorted(proxy)
        out["observed_closing_proxy"]["ensemble_1x2"]=metric(pp(px,"probabilities_1x2"),yy(px))
        out["observed_closing_proxy"]["provider_fair_1x2"]=metric(
          np.array([proxy[x]["fair_probs_1x2"] for x in px]),yy(px))
    for lg in sorted({preds[x]["league"] for x in ids}):
        ms=[x for x in ids if preds[x]["league"]==lg]; out["by_league"][lg]=metric(pp(ms,"probabilities_1x2"),yy(ms))
    times=[utc(preds[x]["generated_at"]) for x in ids]; days=(max(times)-min(times)).days
    out["observation_gate"]={"minimum_matches":1000,"minimum_days":56,"matches_met":len(ids)>=1000,"days_met":days>=56,"production_approval":False}
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise FileExistsError(f"Refusing to overwrite {p}")
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"ok","settled":len(ids),"report":str(p),"production_approval":False},indent=2))

def main():
    p=argparse.ArgumentParser(description="XG18 shadow ledger; no betting or tuning"); s=p.add_subparsers(dest="cmd",required=True)
    q=s.add_parser("predict"); q.add_argument("--history",required=True); q.add_argument("--fixtures",required=True); q.add_argument("--ledger",default="shadow/shadow_ledger.jsonl")
    q.add_argument("--model-module",default="research/xg18-frozen-source/xg18_frozen_model.py"); q.add_argument("--generated-at"); q.set_defaults(fn=predict)
    q=s.add_parser("add-odds"); q.add_argument("--input",required=True); q.add_argument("--ledger",default="shadow/shadow_ledger.jsonl"); q.add_argument("--provider-id",default=""); q.set_defaults(fn=add_odds)
    q=s.add_parser("settle"); q.add_argument("--input",required=True); q.add_argument("--ledger",default="shadow/shadow_ledger.jsonl"); q.set_defaults(fn=settle)
    q=s.add_parser("report"); q.add_argument("--ledger",default="shadow/shadow_ledger.jsonl"); q.add_argument("--output",default="reports/xg18_shadow_report.json"); q.set_defaults(fn=report)
    a=p.parse_args()
    try: a.fn(a)
    except Exception as e: print(f"ERROR: {e}",file=sys.stderr); raise SystemExit(2)
if __name__=="__main__": main()
