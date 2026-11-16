#!/usr/bin/env python3
"""V3 extensions to the leakage-safe XG18 v2 pre-match feature builder."""
from __future__ import annotations
import numpy as np
import pandas as pd
from xg18_v2_features import build_prematch_features as _base

def build_prematch_features(matches: pd.DataFrame) -> pd.DataFrame:
    f=_base(matches).copy()
    for side in ("home","away"):
        for n in (3,5,10):
            gf=f[f"{side}_gf_last{n}"];ga=f[f"{side}_ga_last{n}"]
            xgf=f[f"{side}_xgf_last{n}"];xga=f[f"{side}_xga_last{n}"]
            sf=f[f"{side}_shotsf_last{n}"];sot=f[f"{side}_sotf_last{n}"]
            big=f[f"{side}_bigf_last{n}"]
            f[f"{side}_goal_diff_last{n}"]=gf-ga
            f[f"{side}_xg_diff_last{n}"]=xgf-xga
            f[f"{side}_xg_overperformance_last{n}"]=gf-xgf
            f[f"{side}_shot_accuracy_last{n}"]=sot/(sf+1e-6)
            f[f"{side}_goal_per_shot_last{n}"]=gf/(sf+1e-6)
            f[f"{side}_sot_per_game_last{n}"]=sot
            f[f"{side}_big_chance_rate_last{n}"]=big
            f[f"{side}_points_per_game_last{n}"]=f[f"{side}_points_last{n}"]
        f[f"{side}_form_momentum"]=f[f"{side}_points_last3"]-f[f"{side}_points_last10"]
        f[f"{side}_xg_momentum"]=f[f"{side}_xgf_last3"]-f[f"{side}_xgf_last10"]
        f[f"{side}_defence_momentum"]=f[f"{side}_xga_last10"]-f[f"{side}_xga_last3"]
        f[f"{side}_venue_goal_diff"]=f[f"{side}_gf_venue5"]-f[f"{side}_ga_venue5"]
        f[f"{side}_venue_xg_diff"]=f[f"{side}_xgf_venue5"]-f[f"{side}_xga_venue5"]
    f["form_points_diff_last5"]=f["home_points_last5"]-f["away_points_last5"]
    f["form_goal_diff_last5"]=f["home_goal_diff_last5"]-f["away_goal_diff_last5"]
    f["form_xg_diff_last5"]=f["home_xg_diff_last5"]-f["away_xg_diff_last5"]
    f["form_xg_diff_last3"]=f["home_xg_diff_last3"]-f["away_xg_diff_last3"]
    f["form_shot_accuracy_diff"]=f["home_shot_accuracy_last5"]-f["away_shot_accuracy_last5"]
    f["form_rest_days_diff"]=f["home_rest_days"]-f["away_rest_days"]
    f["form_elo_diff"]=f["home_elo"]-f["away_elo"]
    f["form_momentum_diff"]=f["home_form_momentum"]-f["away_form_momentum"]
    return f.replace([np.inf,-np.inf],np.nan).fillna(0.0)
