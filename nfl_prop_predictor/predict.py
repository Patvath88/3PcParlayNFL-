# file: src/nfl_prop_predictor/predict.py
from __future__ import annotations

from math import erf, sqrt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load

from .features import build_training_frame, load_defenses, load_games, load_schedule
from .utils import TARGETS

def _normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + np.vectorize(lambda v: erf(v / sqrt(2.0)))(x))

def _load_model_bundle(models_dir: Path, target: str):
    mdir = models_dir / target
    if not mdir.exists():
        raise FileNotFoundError(f"Missing model dir: {mdir}")
    model = load(mdir / "model.joblib")
    features = (mdir / "features.txt").read_text().splitlines()
    import json
    sigma = 10.0
    card_path = mdir / "model_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text()); sigma = float(card.get("residual_std", 10.0))
    return model, features, sigma

def predict_schedule(data_dir: str, models_dir: str, schedule_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    data_dir_path = Path(data_dir); models_dir_path = Path(models_dir)
    games = load_games(str(data_dir_path / "games.csv"))
    defenses = load_defenses(str(data_dir_path / "defenses.csv"))
    schedule = load_schedule(schedule_path)
    combined = pd.concat([games, schedule.assign(**{t: np.nan for t in TARGETS if t not in schedule.columns})], ignore_index=True, sort=False)
    feat_all = build_training_frame(combined, defenses)
    key_cols = ["player_id","date","team","opp_team","position","season","week","home_away"]
    feat_sched = feat_all.merge(schedule[key_cols + ["player_name"]], on=key_cols, how="inner")
    preds = schedule.copy()
    for target in TARGETS:
        mdir = models_dir_path / target
        if not mdir.exists(): continue
        model, feat_list, sigma = _load_model_bundle(models_dir_path, target)
        X = feat_sched.reindex(columns=feat_list).fillna(0.0)
        y_hat = model.predict(X)
        preds[f"pred_{target}"] = y_hat
        preds[f"unc_sigma_{target}"] = sigma
        line_col = f"line_{target}"
        if line_col in preds.columns:
            z = (preds[line_col].astype(float) - y_hat) / (sigma if sigma > 1e-6 else 1.0)
            preds[f"prob_over_{target}"] = (1 - _normal_cdf(z)).clip(0, 1)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True); preds.to_csv(output_path, index=False)
    return preds
