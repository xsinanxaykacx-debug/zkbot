#!/usr/bin/env python3
"""Join xg18 pre-match history with all 70 Football-Data odds CSVs.

Only xG matches dated on/before 2024-04-08 are eligible. A match is accepted
only on exact date + mapped league + normalized home/away teams + FT score.
Conflicting duplicates remain unmatched. No final-test rows are inspected.
"""
from __future__ import annotations
import argparse, json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

CUTOFF = pd.Timestamp("2024-04-08")
LEAGUE_TO_CODE = {
    "england premier league":"E0", "premier league":"E0", "premier lig":"E0",
    "england league one":"E2", "league one":"E2", "ingiltere league one":"E2",
    "germany bundesliga":"D1", "bundesliga":"D1",
    "germany 2 bundesliga":"D2", "2 bundesliga":"D2", "2. bundesliga":"D2",
    "spain la liga":"SP1", "la liga":"SP1", "primera division":"SP1",
    "spain segunda division":"SP2", "segunda division":"SP2",
    "italy serie a":"I1", "serie a":"I1",
    "italy serie b":"I2", "serie b":"I2",
    "france ligue 1":"F1", "ligue 1":"F1",
    "france ligue 2":"F2", "ligue 2":"F2",
    "netherlands eredivisie":"N1", "eredivisie":"N1", "holland eredivisie":"N1",
    "portugal liga portugal":"P1", "primeira liga":"P1", "liga portugal":"P1",
    "belgium pro league":"B1", "belgian pro league":"B1", "belcika pro league":"B1",
    "turkiye super lig":"T1", "super lig":"T1", "turkey super lig":"T1",
    "turkiye 1 lig":"T2", "turkey 1 lig":"T2",
}
ALIASES = {
    "manchester united":"man united","man utd":"man united","manchester city":"man city",
    "tottenham hotspur":"tottenham","wolverhampton wanderers":"wolves",
    "afc bournemouth":"bournemouth","leicester city":"leicester",
    "brighton hove albion":"brighton","west ham united":"west ham",
    "newcastle united":"newcastle","leeds united":"leeds",
    "nottingham forest":"nott m forest","nottm forest":"nott m forest",
    "sheffield wednesday":"sheffield weds","bristol rovers":"bristol rvs",
    "celta vigo":"celta","espanyol":"espanol","athletic club":"ath bilbao",
    "atletico madrid":"ath madrid","real valladolid":"valladolid",
    "real sociedad":"sociedad","real betis":"betis",
    "bayern munchen":"bayern munich","eintracht frankfurt":"ein frankfurt",
    "borussia dortmund":"dortmund","borussia monchengladbach":"m gladbach",
    "vfb stuttgart":"stuttgart","bayer leverkusen":"leverkusen",
    "1 fc koln":"fc koln","hertha bsc":"hertha","mainz 05":"mainz",
    "paris saint germain":"paris sg","hellas verona":"verona",
    "sc heerenveen":"heerenveen","fortuna sittard":"for sittard",
    "rkc waalwijk":"waalwijk","fc groningen":"groningen",
    "fc volendam":"volendam","nec nijmegen":"nijmegen","fc utrecht":"utrecht",
    "fc twente":"twente","fc porto":"porto","sporting cp":"sporting",
    "sp braga":"braga","vitoria de guimaraes":"guimaraes",
    "pacos de ferreira":"pacos ferreira","casa pia ac":"casa pia",
    "standard liege":"standard","sporting charleroi":"charleroi",
    "zulte waregem":"waregem","union st gilloise":"st gilloise",
    "kv mechelen":"mechelen","royal antwerp":"antwerp",
    "fatih karagumruk":"karagumruk","adana demirspor":"ad demirspor",
    "basaksehir":"buyuksehyr","gaziantep fk":"gaziantep",
    "caykur rizespor":"rizespor","ankaragucu":"mke ankaragucu",
}
def norm(v):
    if v is None or pd.isna(v): return ""
    s=unicodedata.normalize("NFKD",str(v).strip().casefold())
    s="".join(c for c in s if not unicodedata.combining(c)).replace("ı","i")
    return " ".join(re.sub(r"[^a-z0-9]+"," ",s).split())
def team(v):
    s=norm(v); return ALIASES.get(s,s)
def read_csv(p):
    try: return pd.read_csv(p,encoding="utf-8-sig",low_memory=False)
    except UnicodeDecodeError: return pd.read_csv(p,encoding="cp1252",low_memory=False)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--xg",type=Path,default=Path("xg18.csv"))
    ap.add_argument("--odds-dir",type=Path,default=Path("."))
    ap.add_argument("--out",type=Path,default=Path("artifacts/xg18_odds"))
    a=ap.parse_args()
    paths=sorted(p for p in a.odds_dir.glob("*.csv") if re.match(r"^(B1|D1|D2|E0|E2|F1|F2|I1|I2|N1|P1|SP1|SP2|T1)( \(\d+\))?\.csv$",p.name,re.I))
    if len(paths)!=70: raise RuntimeError(f"Expected exactly 70 odds CSVs; found {len(paths)}")
    records=[]; inventory=[]; odds_columns=set(); file_rows=0
    for p in paths:
        d=read_csv(p); file_rows+=len(d)
        required={"Date","HomeTeam","AwayTeam","FTHG","FTAG"}
        if not required.issubset(d.columns): raise ValueError(f"{p.name}: missing {sorted(required-set(d.columns))}")
        code=str(d["Div"].dropna().iloc[0]).strip() if "Div" in d and d["Div"].notna().any() else p.stem.split(" ")[0]
        d["_date"]=pd.to_datetime(d["Date"],dayfirst=True,errors="coerce").dt.normalize()
        d["_home"]=d["HomeTeam"].map(team); d["_away"]=d["AwayTeam"].map(team)
        d["_hg"]=pd.to_numeric(d["FTHG"],errors="coerce"); d["_ag"]=pd.to_numeric(d["FTAG"],errors="coerce")
        odds=[c for c in d.columns if c not in required|{"Div","Date","Time","HomeTeam","AwayTeam","FTHG","FTAG","HTHG","HTAG","HTR","Referee","_date","_home","_away","_hg","_ag"}]
        odds_columns.update(odds)
        valid=d["_date"].notna()
        inventory.append({"file":p.name,"league_code":code,"rows":len(d),"valid_date_rows":int(valid.sum()),"date_min":str(d.loc[valid,"_date"].min().date()) if valid.any() else None,"date_max":str(d.loc[valid,"_date"].max().date()) if valid.any() else None})
        for _,r in d.loc[valid].iterrows():
            records.append({"date":r["_date"],"code":code,"home":r["_home"],"away":r["_away"],"hg":r["_hg"],"ag":r["_ag"],"file":p.name,"row":r})
    x=read_csv(a.xg)
    need={"Date","League","HomeTeam","AwayTeam","FTHG","FTAG","Home_xG","Away_xG"}
    if not need.issubset(x.columns): raise ValueError(f"xg18 missing {sorted(need-set(x.columns))}")
    x["_date"]=pd.to_datetime(x["Date"],errors="coerce",utc=True).dt.tz_convert(None).dt.normalize()
    eligible=x.loc[x["_date"].notna() & (x["_date"]<=CUTOFF)].copy()
    by_key=defaultdict(list)
    for r in records: by_key[(r["date"],r["code"],r["home"],r["away"])].append(r)
    out=[]; status=Counter(); by_league=defaultdict(Counter)
    for idx,r in eligible.iterrows():
        league=norm(r["League"]); code=LEAGUE_TO_CODE.get(league)
        key=(r["_date"],code,team(r["HomeTeam"]),team(r["AwayTeam"]))
        cand=by_key.get(key,[]) if code else []
        hg=pd.to_numeric(r["FTHG"],errors="coerce"); ag=pd.to_numeric(r["FTAG"],errors="coerce")
        scored=[z for z in cand if pd.notna(hg) and pd.notna(ag) and hg==z["hg"] and ag==z["ag"]]
        chosen=None
        if not code: st="league_unmapped"
        elif len(scored)==1: chosen=scored[0]; st="matched"
        elif len(scored)>1:
            sigs={tuple(str(z["row"].get(c,"")) for c in sorted(odds_columns)) for z in scored}
            if len(sigs)==1: chosen=scored[0]; st="matched_identical_duplicate"
            else: st="ambiguous_conflicting_odds"
        elif cand: st="score_mismatch"
        else: st="no_candidate"
        status[st]+=1; by_league[str(r["League"])][st]+=1
        row={"xg_source_row":int(idx),"MatchID":r.get("MatchID"),"Date":str(r["_date"].date()),"League":r["League"],"HomeTeam":r["HomeTeam"],"AwayTeam":r["AwayTeam"],"FTHG":hg,"FTAG":ag,"Home_xG":r["Home_xG"],"Away_xG":r["Away_xG"],"odds_match_status":st,"odds_source_file":chosen["file"] if chosen else None}
        if chosen:
            for c in odds_columns: row["odds_"+c]=chosen["row"].get(c)
        else:
            for c in odds_columns: row["odds_"+c]=pd.NA
        out.append(row)
    joined=pd.DataFrame(out)
    matched=joined[joined.odds_match_status.str.startswith("matched")]
    def cov(cols):
        cols=[c for c in cols if c in matched]
        ok=matched[cols].notna().all(axis=1) if cols else pd.Series(False,index=matched.index)
        return {"complete_rows":int(ok.sum()),"matched_rows":int(len(matched)),"coverage_pct":round(100*ok.mean(),2) if len(matched) else 0.0}
    summary={"xg18_rows":len(x),"eligible_xg_rows_through_2024_04_08":len(eligible),"xg_rows_after_cutoff_excluded":int((x["_date"]>CUTOFF).sum()),"odds_files":len(paths),"odds_source_rows":file_rows,"odds_valid_date_rows":len(records),"joined_rows":len(joined),"matched_rows":len(matched),"unmatched_rows":int(len(joined)-len(matched)),"status_counts":dict(status),"by_league":{k:{"total":sum(v.values()),"matched":v["matched"]+v["matched_identical_duplicate"],"statuses":dict(v)} for k,v in sorted(by_league.items())},"market_coverage_on_matched_rows":{"1x2_avg":cov(["odds_AvgH","odds_AvgD","odds_AvgA"]),"1x2_b365":cov(["odds_B365H","odds_B365D","odds_B365A"]),"ou25_avg":cov(["odds_Avg>2.5","odds_Avg<2.5"]),"ou25_b365":cov(["odds_B365>2.5","odds_B365<2.5"])},"league_csv_inventory":inventory,"final_test_rows_used":0,"policy":"exact date + mapped league + normalized home/away + FT score; ambiguous conflicts unmatched; odds missing remain null; final-test xG dates excluded before matching"}
    a.out.mkdir(parents=True,exist_ok=True)
    joined.to_csv(a.out/"xg18_with_all_league_odds.csv",index=False,encoding="utf-8")
    (a.out/"merge_report.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
