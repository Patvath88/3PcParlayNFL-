# file: src/nfl_prop_predictor/predict.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import norm

from .features import (
    build_training_frame,
    load_defenses,
    load_games,
    load_schedule,
)
from .utils import TARGETS


def _load_model_bundle(models_dir: Path, target: str):
    """Load trained model, its feature list, and residual std (σ) for a target."""
    mdir = models_dir / target
    if not mdir.exists():
        raise FileNotFoundError(f"Model directory not found for target '{target}': {mdir}")
    model = load(mdir / "model.joblib")
    features = (mdir / "features.txt").read_text().splitlines()
    import json

    card_path = mdir / "model_card.json"
    sigma = 10.0
    if card_path.exists():
        card = json.loads(card_path.read_text())
        sigma = float(card.get("residual_std", 10.0))
    return model, features, sigma


def predict_schedule(
    data_dir: str,
    models_dir: str,
    schedule_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build predictions for the upcoming schedule using previously trained models.
    Returns a DataFrame (and optionally writes CSV if output_path is provided).
    """
    data_dir_path = Path(data_dir)
    models_dir_path = Path(models_dir)

    # Load inputs
    games = load_games(str(data_dir_path / "games.csv"))
    defenses = load_defenses(str(data_dir_path / "defenses.csv"))
    schedule = load_schedule(schedule_path)

    # Concatenate to let schedule inherit rolling/lag context from games
    # Create placeholder target cols for schedule so featurizer runs uniformly
    sched_targets = {t: np.nan for t in TARGETS if t not in schedule.columns}
    combined = pd.concat(
        [games, schedule.assign(**sched_targets)],
        ignore_index=True,
        sort=False,
    )

    feat_all = build_training_frame(combined, defenses)

    # Extract featurized rows for the schedule (match by keys)
    key_cols = ["player_id", "date", "team", "opp_team", "position", "season", "week", "home_away"]
    feat_sched = feat_all.merge(schedule[key_cols + ["player_name"]], on=key_cols, how="inner")

    # Preserve original schedule row order
    feat_sched["_merge_order"] = (
        schedule.reset_index()[key_cols]
        .merge(feat_sched[key_cols], on=key_cols, how="left")
        .reset_index().index
    )
    feat_sched = feat_sched.sort_values("_merge_order").drop(columns=["_merge_order"])

    preds = schedule.copy()

    for target in TARGETS:
        mdir = models_dir_path / target
        if not mdir.exists():
            # No model trained for this target; skip quietly
            continue

        model, feat_list, sigma = _load_model_bundle(models_dir_path, target)
        X = feat_sched.reindex(columns=feat_list).fillna(0.0)
        y_hat = model.predict(X)

        preds[f"pred_{target}"] = y_hat
        preds[f"unc_sigma_{target}"] = sigma

        # If schedule already has a line for this target, compute P(over)
        line_col = f"line_{target}"
        if line_col in preds.columns:
            z = (preds[line_col].astype(float) - y_hat) / (sigma if sigma > 1e-6 else 1.0)
            preds[f"prob_over_{target}"] = (1 - norm.cdf(z)).clip(0, 1)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(output_path, index=False)

    return preds


def line_probability(preds_csv: str, out: Optional[str] = None) -> pd.DataFrame:
    """
    Given a predictions CSV (with pred_* and unc_sigma_*), compute P(over)
    for any available line_* columns not already present, then optionally write out.
    """
    df = pd.read_csv(preds_csv)

    for target in TARGETS:
        pred_col = f"pred_{target}"
        sigma_col = f"unc_sigma_{target}"
        line_col = f"line_{target}"
        prob_col = f"prob_over_{target}"

        if all(c in df.columns for c in [pred_col, sigma_col, line_col]) and prob_col not in df.columns:
            # guard against sigma=0
            sigma = df[sigma_col].replace(0, 1.0).astype(float)
            z = (df[line_col].astype(float) - df[pred_col].astype(float)) / sigma
            df[prob_col] = (1 - norm.cdf(z)).clip(0, 1)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    return df
