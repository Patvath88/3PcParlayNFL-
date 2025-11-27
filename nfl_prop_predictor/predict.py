from future import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from joblib import load
from scipy.stats import norm
from .features import load_games, load_defenses, load_schedule, build_training_frame
from .utils import TARGETS

def _load_model_bundle(models_dir: Path, target: str):
mdir = models_dir / target
model = load(mdir / "model.joblib")
features = (mdir / "features.txt").read_text().splitlines()
import json
card = json.loads((mdir / "model_card.json").read_text())
sigma = float(card.get("residual_std", 10.0))
return model, features, sigma

def predict_schedule(data_dir: str, models_dir: str, schedule_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
data_dir = Path(data_dir)
models_dir = Path(models_dir)

games = load_games(str(data_dir / "games.csv"))
defenses = load_defenses(str(data_dir / "defenses.csv"))
schedule = load_schedule(schedule_path)

combined = pd.concat([games, schedule.assign(**{t: np.nan for t in TARGETS})], ignore_index=True, sort=False)
feat = build_training_frame(combined, defenses)

mask = feat["date"].isin(schedule["date"]) & feat["player_id"].isin(schedule["player_id"])
feat_sched = feat.loc[mask].copy()

# Preserve schedule ordering
key_cols = ["player_id","date","team","opp_team","position","season","week","home_away"]
feat_sched = feat_sched.merge(
    schedule[key_cols + ["player_name"]], on=key_cols, how="right"
)

preds = schedule.copy()
for target in TARGETS:
    mdir = models_dir / target
    if not mdir.exists():
        continue
    model, feat_list, sigma = _load_model_bundle(models_dir, target)
    X = feat_sched.reindex(columns=feat_list).fillna(0.0)
    y_hat = model.predict(X)
    preds[f"pred_{target}"] = y_hat
    preds[f"unc_sigma_{target}"] = sigma

    line_col = f"line_{target}"
    if line_col in preds.columns:
        z = (preds[line_col].astype(float) - y_hat) / (sigma if sigma > 1e-6 else 1.0)
        preds[f"prob_over_{target}"] = (1 - norm.cdf(z)).clip(0, 1)

if output_path:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path, index=False)
return preds


def line_probability(preds_csv: str, out: Optional[str] = None) -> pd.DataFrame:
df = pd.read_csv(preds_csv)
for target in TARGETS:
pred_col = f"pred_{target}"
sigma_col = f"unc_sigma_{target}"
line_col = f"line_{target}"
prob_col = f"prob_over_{target}"
if all(c in df.columns for c in [pred_col, sigma_col, line_col]) and prob_col not in df.columns:
z = (df[line_col].astype(float) - df[pred_col].astype(float)) / df[sigma_col].replace(0, 1.0)
df[prob_col] = (1 - norm.cdf(z)).clip(0, 1)
if out:
Path(out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
return df
