from future import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from .features import load_games, load_defenses, build_training_frame, feature_targets
from .data_models import TrainConfig, ModelCard
from .utils import TARGETS
from joblib import dump

def _time_aware_split(df: pd.DataFrame, val_size: float):
df = df.sort_values("date")
split_idx = int(len(df) * (1 - val_size))
return df.iloc[:split_idx], df.iloc[split_idx:]

def train_models(data_dir: str, models_dir: str, config: TrainConfig = TrainConfig()):
data_dir = Path(data_dir)
models_dir = Path(models_dir)
models_dir.mkdir(parents=True, exist_ok=True)
games = load_games(str(data_dir / "games.csv"))
defenses = load_defenses(str(data_dir / "defenses.csv"))
df = build_training_frame(games, defenses)

results = []
for target in TARGETS:
    rows = df[df[f"{target}_lag1"].notna()].copy()
    if rows.empty:
        continue
    train_df, val_df = _time_aware_split(rows, config.validation_size)
    X_tr, y_tr = feature_targets(train_df, target)
    X_va, y_va = feature_targets(val_df, target)

    model = HistGradientBoostingRegressor(random_state=config.random_state)
    model.fit(X_tr.fillna(0.0), y_tr)

    va_pred = model.predict(X_va.fillna(0.0))
    rmse = mean_squared_error(y_va, va_pred, squared=False)
    r2 = r2_score(y_va, va_pred)
    resid_std = float(np.std(y_va - va_pred, ddof=1)) if len(y_va) > 1 else 10.0  # why: calibrates P(over)

    try:
        pi = permutation_importance(model, X_va.fillna(0.0), y_va, n_repeats=5, random_state=config.random_state)
        importances = sorted(zip(X_va.columns, pi.importances_mean), key=lambda x: -x[1])[:10]
        top_feats = ", ".join([f"{k}:{v:.3f}" for k, v in importances[:5]])
    except Exception:
        top_feats = "n/a"

    card = ModelCard(
        target=target,
        features=list(X_tr.columns),
        residual_std=resid_std,
        train_rows=int(len(X_tr)),
        val_rows=int(len(X_va)),
        notes=f"RMSE={rmse:.2f}, R2={r2:.3f}, top_feats={top_feats}"
    )

    out_dir = models_dir / target
    out_dir.mkdir(exist_ok=True)
    dump(model, out_dir / "model.joblib")
    (out_dir / "features.txt").write_text("\n".join(card.features))
    (out_dir / "model_card.json").write_text(card.model_dump_json(indent=2))
    results.append((target, rmse, r2, resid_std))

summary = pd.DataFrame(results, columns=["target", "rmse", "r2", "resid_std"]) if results else pd.DataFrame(columns=["target","rmse","r2","resid_std"])
summary.to_csv(models_dir / "training_summary.csv", index=False)
return summary
