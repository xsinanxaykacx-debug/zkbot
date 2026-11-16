import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import pandas as pd
from xg18_v2_features import build_prematch_features

BASE={"League":"L1","Season":"2025/2026","HT_Home_Goals":0,"HT_Away_Goals":0,"Home_xG":1.2,"Away_xG":0.8,
 "Home_Shots":10,"Away_Shots":8,"Home_ShotsOnTarget":4,"Away_ShotsOnTarget":3,"Home_BigChance":1,"Away_BigChance":0,
 "Home_Corners":4,"Away_Corners":3,"Home_Poss":55,"Away_Poss":45,"Home_RedCards":0,"Away_RedCards":0}
def row(home,away,date,hg,ag):
 d=BASE.copy();d.update({"HomeTeam":home,"AwayTeam":away,"Date":date,"FTHG":hg,"FTAG":ag});return d

def test_same_day_results_cannot_change_same_day_features():
 rows=[row("A","B","2025-01-01T10:00:00Z",2,0),row("C","D","2025-01-01T18:00:00Z",1,1),row("A","C","2025-01-02T18:00:00Z",0,0)]
 changed=[dict(x) for x in rows];changed[0]["FTHG"]=0;changed[0]["FTAG"]=2
 a=build_prematch_features(pd.DataFrame(rows));b=build_prematch_features(pd.DataFrame(changed))
 pd.testing.assert_series_equal(a.loc[0],b.loc[0]);pd.testing.assert_series_equal(a.loc[1],b.loc[1])
 assert a.loc[2,"home_gf_last3"]!=b.loc[2,"home_gf_last3"]
 assert a.loc[2,"home_elo"]!=b.loc[2,"home_elo"]
