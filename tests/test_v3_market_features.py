import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
import pandas as pd
from xg18_v3_features import build_prematch_features

def _base():
    return {"League":"L","Season":"S","HomeTeam":"A","AwayTeam":"B",
      "Date":"2025-01-01T10:00:00Z","FTHG":1,"FTAG":0,
      "Home_xG":1.1,"Away_xG":0.9,"Home_Shots":10,"Away_Shots":8,
      "Home_ShotsOnTarget":4,"Away_ShotsOnTarget":3,
      "Home_BigChance":1,"Away_BigChance":0}

def test_market_odds_become_normalised_prematch_probabilities():
    row=_base()|{"odds_AvgH":2.0,"odds_AvgD":3.0,"odds_AvgA":4.0,
      "odds_Avg>2.5":1.8,"odds_Avg<2.5":2.0}
    f=build_prematch_features(pd.DataFrame([row]))
    p=f.loc[0,["market_avg_p_home","market_avg_p_draw","market_avg_p_away"]].to_numpy(float)
    expected=np.array([1/2,1/3,1/4]);expected/=expected.sum()
    np.testing.assert_allclose(p,expected,rtol=1e-10)
    assert f.loc[0,"market_avg_1x2_present"]==1
    assert f.loc[0,"market_avg_ou25_present"]==1

def test_missing_odds_use_neutral_prior_and_presence_flag():
    f=build_prematch_features(pd.DataFrame([_base()]))
    np.testing.assert_allclose(f.loc[0,["market_avg_p_home","market_avg_p_draw","market_avg_p_away"]].to_numpy(float),[1/3]*3)
    assert f.loc[0,"market_avg_1x2_present"]==0
    assert np.isfinite(f.select_dtypes("number").to_numpy()).all()
