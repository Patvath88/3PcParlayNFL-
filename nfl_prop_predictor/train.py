# file: src/nfl_prop_predictor/train.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

from .data_models import ModelCard, TrainConfig
from .features import build_training_frame, feature_targets, load_defenses, load_games
from .utils import TARGETS


def _time_aware_split(df: pd.DataFrame, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split to avoid leakage."""
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - val_size))
    split_idx = max(1, min(split_idx, len(df) - 1))  # why: ensure non-empty splits
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_models(
    data_dir: str,
    models_dir: str,
    config: TrainConfig = TrainConfig(),
) -> pd.DataFrame:
    """Train one model per target; persist model, features, and model card."""
    data_dir_path = Path(data_dir)
    models_dir_path = Path(models_dir)
    models_dir_path.mkdir(parents=True, exist_ok=True)

    games = load_games(str(data_dir_path / "games.csv"))
    defenses = load_defenses(str(data_dir_path / "defenses.csv"))
    df = build_training_frame(games, defenses)

    results = []

    for target in TARGETS:
        # Require at least a previous game (lag1) to form features
        rows = df[df.get(f"{target}_lag1").notna()].copy()
        if rows.empty or len(rows) < 20:
            # Skip targets without enough history
            continue

        train_df, val_df = _time_aware_split(rows, config.validation_size)
        X_tr, y_tr = feature_targets(train_df, target)
        X_va, y_va = feature_targets(val_df, target)

        # Train
        model = HistGradientBoostingRegressor(random_state=config.random_state)
        model.fit(X_tr.fillna(0.0), y_tr)

        # Validate
        va_pred = model.predict(X_va.fillna(0.0))
        rmse = float(mean_squared_error(y_va, va_pred, squared=False))
        r2 = float(r2_score(y_va, va_pred))
        resid_std = float(np.std(y_va - va_pred, ddof=1)) if len(y_va) > 1 else 10.0  # why: used for prop probability

        # Permutation importances (best-effort)
        try:
            pi = permutation_importance(
                model, X_va.fillna(0.0), y_va, n_repeats=5, random_state=config.random_state
            )
            top_feats = ", ".join(
                f"{name}:{imp:.3f}"
                for name, imp in sorted(
                    zip(X_va.columns.tolist(), pi.importances_mean.tolist()), key=lambda x: -x[1]
                )[:5]
            )
        except Exception:
            top_feats = "n/a"

        # Persist artifacts
        out_dir = models_dir_path / target
        out_dir.mkdir(parents=True, exist_ok=True)
        dump(model, out_dir / "model.joblib")
        (out_dir / "features.txt").write_text("\n".join(X_tr.columns))
        (out_dir / "model_card.json").write_text(
            ModelCard(
                target=target,
                features=list(X_tr.columns),
                residual_std=resid_std,
                train_rows=int(len(X_tr)),
                val_rows=int(len(X_va)),
                notes=f"RMSE={rmse:.2f}, R2={r2:.3f}, top_feats={top_feats}",
            ).model_dump_json(indent=2)
        )

        results.append((target, rmse, r2, resid_std))

    summary = (
        pd.DataFrame(results, columns=["target", "rmse", "r2", "resid_std"])
        if results
        else pd.DataFrame(columns=["target", "rmse", "r2", "resid_std"])
    )
    summary.to_csv(models_dir_path / "training_summary.csv", index=False)
    return summary


# Optional: allow `python -m nfl_prop_predictor.train --data-dir ... --models-dir ...`
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(validation_size=args.validation_size, random_state=args.random_state)
    out = train_models(args.data_dir, args.models_dir, cfg)
    print(out.to_string(index=False))
