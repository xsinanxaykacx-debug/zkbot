import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import pandas as pd
from xg18_v3_features import build_prematch_features

def test_same_day_results_do_not_leak_into_same_day_features():
    base={"League":"L","Season":"S","Home_xG":1.1,"Away_xG":.9,"Home_Shots":10,"Away_Shots":8,
      "Home_ShotsOnTarget":4,"Away_ShotsOnTarget":3,"Home_BigChance":1,"Away_BigChance":0}
    rows=[]
    for h,a,d,hg,ag in [("A","B","2025-01-01T10:00:00Z",2,0),("C","D","2025-01-01T18:00:00Z",1,1),("A","C","2025-01-02T18:00:00Z",0,0)]:
        r=base.copy();r.update({"HomeTeam":h,"AwayTeam":a,"Date":d,"FTHG":hg,"FTAG":ag});rows.append(r)
    changed=[dict(x) for x in rows];changed[0]["FTHG"]=0;changed[0]["FTAG"]=2
    x=build_prematch_features(pd.DataFrame(rows));z=build_prematch_features(pd.DataFrame(changed))
    pd.testing.assert_series_equal(x.loc[0],z.loc[0]);pd.testing.assert_series_equal(x.loc[1],z.loc[1])
    assert x.loc[2,"home_gf_last3"]!=z.loc[2,"home_gf_last3"]
    assert "form_xg_diff_last5" in x.columns

def test_features_are_finite():
    base={"League":"L","Season":"S","Home_xG":1.1,"Away_xG":.9,"Home_Shots":10,"Away_Shots":8,
      "Home_ShotsOnTarget":4,"Away_ShotsOnTarget":3,"Home_BigChance":1,"Away_BigChance":0}
    r=base|{"HomeTeam":"A","AwayTeam":"B","Date":"2025-01-01T10:00:00Z","FTHG":1,"FTAG":0}
    x=build_prematch_features(pd.DataFrame([r]))
    assert x.select_dtypes("number").notna().all().all()
