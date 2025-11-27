"""
Streamlit UI for NFL Prop Predictor with integrated free data fetch (no API key).
Run:
  PYTHONPATH=src streamlit run app.py
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="NFL Prop Predictor", layout="wide")

# --- Local package imports (requires PYTHONPATH=src) ---
try:
    from nfl_prop_predictor.train import train_models
    from nfl_prop_predictor.predict import predict_schedule
    from nfl_prop_predictor.utils import TARGETS
except Exception as e:
    st.error(
        f"Import error: {e}\n\n"
        "Run with `PYTHONPATH=src streamlit run app.py` or install the package."
    )
    TARGETS = ["pass_yards","rush_yards","rec_yards","receptions","pass_td","rush_td","rec_td"]

# --- Free data source: nflverse via nflreadpy (no key required) ---
import nflreadpy as nfl  # pulls public, CC-BY nflverse datasets

# ---------- Helpers: Build datasets ----------
def _pick_player_id(df: pd.DataFrame) -> str:
    """Pick the best available ID column name from nflverse tables."""
    for c in ["player_id", "gsis_id", "player_gsis_id", "nfl_id", "pfr_id", "esb_id"]:
        if c in df.columns:
            return c
    raise RuntimeError("No player id column found in source data.")

def _position_from_row(row: pd.Series) -> str:
    pos = str(row.get("position") or row.get("player_position") or "").upper()
    return pos if pos in {"QB","RB","WR","TE"} else pos

def _build_games(seasons: List[int]) -> pd.DataFrame:
    ps = nfl.load_player_stats(seasons).to_pandas()
    pid = _pick_player_id(ps)
    mapping = {
        pid: "player_id",
        "player_name": "player_name",
        "recent_team": "team",
        "opponent_team": "opp_team",
        "game_date": "date",
        "season": "season",
        "week": "week",
        "home_away": "home_away",
        # targets
        "passing_yards": "pass_yards",
        "rushing_yards": "rush_yards",
        "receiving_yards": "rec_yards",
        "receptions": "receptions",
        "passing_tds": "pass_td",
        "rushing_tds": "rush_td",
        "receiving_tds": "rec_td",
        # optional context
        "team_score": "team_points",
        "opponent_score": "opp_points",
        "spread_line": "spread",
        "total_line": "total",
        "targets": "targets",
        "rushing_attempts": "rush_att",
        "passing_attempts": "pass_att",
        "offense_snapshare": "snap_pct",
    }
    mapping = {src: dst for src, dst in mapping.items() if src in ps.columns}
    df = ps.rename(columns=mapping)
    req = ["player_id","player_name","team","opp_team","date","season","week","home_away"]
    for col in req:
        if col not in df.columns:
            raise RuntimeError(f"games.csv missing required column from source: {col}")
    if "position" not in df.columns:
        df["position"] = df.apply(_position_from_row, axis=1)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["team"] = df["team"].str.upper()
    df["opp_team"] = df["opp_team"].str.upper()
    df["home_away"] = df["home_away"].fillna("").str.upper().map({"HOME":"H","AWAY":"A","H":"H","A":"A"})
    for t in TARGETS:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce")
        else:
            df[t] = np.nan
    for c in ["team_points","opp_points","spread","total","snap_pct","targets","rush_att","pass_att"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = list({*req, "position", *TARGETS, "team_points","opp_points","spread","total","snap_pct","targets","rush_att","pass_att"})
    return df[keep].sort_values(["player_id","date"])

def _build_defenses(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in TARGETS:
        if metric not in games.columns:
            continue
        tmp = (
            games.groupby(["season","opp_team","position"], as_index=False)[metric]
            .sum()
            .rename(columns={"opp_team":"team", metric:"sum_val"})
        )
        counts = games.groupby(["season","opp_team","position"], as_index=False).size().rename(columns={"opp_team":"team","size":"n"})
        tmp = tmp.merge(counts, on=["season","team","position"], how="left")
        tmp["allowed_per_game"] = tmp["sum_val"] / tmp["n"].replace(0, np.nan)
        tmp["metric"] = metric
        rows.append(tmp[["season","team","position","metric","allowed_per_game"]])
    out = pd.concat(rows, ignore_index=True)
    out["rank"] = out.groupby(["season","position","metric"])["allowed_per_game"].rank(method="min", ascending=True).astype(int)
    return out.sort_values(["season","position","metric","rank"])

def _build_player_schedule_next_week(seasons: List[int]) -> pd.DataFrame:
    season_latest = max(seasons)
    sched = nfl.load_schedules(season_latest).to_pandas()
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    upcoming = sched[sched["gameday"] >= pd.Timestamp("today").normalize()].sort_values("gameday")
    next_wk = int(upcoming["week"].iloc[0]) if not upcoming.empty else int(sched["week"].max())
    wk = sched[sched["week"] == next_wk].copy()
    wk["date"] = wk["gameday"].dt.date

    # Player universe: active players with at least one game this season
    stats = nfl.load_player_stats(season_latest).to_pandas()
    pid = _pick_player_id(stats)
    # Keep skilled positions only
    pos_col = "position" if "position" in stats.columns else ("player_position" if "player_position" in stats.columns else None)
    if pos_col is None:
        stats["__pos__"] = ""
        pos_col = "__pos__"
    stats["__pos_norm__"] = stats[pos_col].astype(str).str.upper()
    skilled = stats[stats["__pos_norm__"].isin(["QB","RB","WR","TE"])].copy()
    skilled = skilled.rename(columns={pid:"player_id"})
    players = skilled[["player_id","player_name","recent_team",pos_col]].dropna().drop_duplicates()
    players = players.rename(columns={"recent_team":"team", pos_col:"position"})
    players["team"] = players["team"].str.upper()
    players["position"] = players["position"].astype(str).str.upper()

    # Build player-level schedule rows by joining team matchups
    def team_rows(frame, team_col, opp_col, hoa):
        df = frame[[team_col, opp_col, "season", "week", "date"]].rename(columns={team_col:"team", opp_col:"opp_team"}).copy()
        df["home_away"] = hoa
        return df

    games_teams = pd.concat(
        [
            team_rows(wk, "home_team", "away_team", "H"),
            team_rows(wk, "away_team", "home_team", "A"),
        ],
        ignore_index=True,
    ).drop_duplicates()

    sched_players = players.merge(games_teams, on="team", how="inner")
    sched_players = sched_players[["player_id","player_name","team","position","opp_team","date","season","week","home_away"]].drop_duplicates()
    # Basic sanity: limit to players who actually logged a game this season
    return sched_players.sort_values(["team","position","player_name"])

@st.cache_data(show_spinner=True)
def fetch_and_build(out_dir: str, seasons_text: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch free data and build the three CSVs."""
    seasons = _parse_seasons(seasons_text)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    games = _build_games(seasons)
    defenses = _build_defenses(games)
    schedule = _build_player_schedule_next_week(seasons)
    games.to_csv(out / "games.csv", index=False)
    defenses.to_csv(out / "defenses.csv", index=False)
    schedule.to_csv(out / "schedule.csv", index=False)
    return games, defenses, schedule

def _parse_seasons(spec: str) -> List[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]

def _write_temp_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name

def _predict_from_disk(data_dir: str, models_dir: str, schedule_csv_path: str) -> pd.DataFrame:
    return predict_schedule(data_dir=data_dir, models_dir=models_dir, schedule_path=schedule_csv_path, output_path=None)

# ---------- UI ----------
st.title("🏈 NFL Prop Predictor — Zero-Upload Edition")

with st.sidebar:
    st.markdown("### Settings")
    data_dir = st.text_input("Data directory", "./data")
    models_dir = st.text_input("Models directory", "./models")
    seasons_text = st.text_input("Seasons (e.g., 2021-2025 or 2023,2024,2025)", "2022-2025")
    st.caption("Data source: nflverse via nflreadpy (no API key).")
    st.divider()
    c1, c2, c3 = st.columns(3)
    btn_fetch = c1.button("⬇️ Fetch Data")
    btn_train = c2.button("🚀 Train Models")
    btn_predict = c3.button("🔮 Predict")

# FETCH
if btn_fetch:
    with st.spinner("Fetching and building datasets..."):
        try:
            games, defenses, schedule = fetch_and_build(data_dir, seasons_text)
            st.success(f"Data ready — games:{len(games):,} defenses:{len(defenses):,} schedule rows:{len(schedule):,}")
            st.dataframe(schedule.head(50), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Fetch failed: {e}")

# TRAIN
if btn_train:
    with st.spinner("Training models..."):
        try:
            summary = train_models(data_dir=data_dir, models_dir=models_dir)
            st.success("Training complete")
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.download_button(
                "Download training_summary.csv",
                summary.to_csv(index=False).encode("utf-8"),
                file_name="training_summary.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Training failed: {e}")

# PREDICT
if btn_predict:
    schedule_path = str(Path(data_dir) / "schedule.csv")
    if not Path(schedule_path).exists():
        st.warning("No schedule.csv found. Click 'Fetch Data' first.")
    else:
        with st.spinner("Predicting..."):
            try:
                preds = _predict_from_disk(data_dir=data_dir, models_dir=models_dir, schedule_csv_path=schedule_path)
                # Quick global line helper
                st.subheader("Predictions")
                gcol1, gcol2 = st.columns(2)
                tgt = gcol1.selectbox("Prop target", TARGETS, index=2)
                line = gcol2.number_input("Global line (optional)", value=0.0, step=0.5)
                if line > 0:
                    pred_col = f"pred_{tgt}"
                    sigma_col = f"unc_sigma_{tgt}"
                    if pred_col in preds.columns and sigma_col in preds.columns:
                        z = (float(line) - preds[pred_col].astype(float)) / preds[sigma_col].replace(0, 1.0)
                        preds[f"prob_over_{tgt}_global"] = (1 - norm.cdf(z)).clip(0, 1)
                st.dataframe(preds, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download predictions.csv",
                    preds.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.caption("Tip: You can re-run Fetch with a different season range to refresh the data.")

# file: src/nfl_prop_predictor/utils.py
from __future__ import annotations
from typing import List, Optional
import pandas as pd

TEAM_ALIASES = {"JAX":"JAC","WSH":"WAS","LA":"LAR"}

def normalize_team(team: Optional[str]) -> Optional[str]:
    if team is None or (isinstance(team, float) and pd.isna(team)): return None
    t = str(team).strip().upper()
    return TEAM_ALIASES.get(t, t)

def normalize_position(pos: Optional[str]) -> Optional[str]:
    if pos is None or (isinstance(pos, float) and pd.isna(pos)): return None
    p = str(pos).strip().upper()
    return p if p in {"QB","RB","WR","TE"} else p

def parse_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df

def ensure_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing: raise ValueError(f"{name} missing columns: {missing}")

def add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_id","date"]).copy()
    prev = out.groupby("player_id")["date"].shift(1)
    d0 = pd.to_datetime(out["date"], errors="coerce")
    d1 = pd.to_datetime(prev, errors="coerce")
    out["days_rest"] = (d0 - d1).dt.days.fillna(7).clip(lower=0, upper=30)
    return out

TARGETS: List[str] = ["pass_yards","rush_yards","rec_yards","receptions","pass_td","rush_td","rec_td"]

# file: src/nfl_prop_predictor/data_models.py
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel

class TrainConfig(BaseModel):
    validation_size: float = 0.15
    min_games_for_features: int = 5
    random_state: int = 42

class ModelCard(BaseModel):
    target: str
    version: str = "1.0"
    features: List[str]
    residual_std: float
    train_rows: int
    val_rows: int
    notes: Optional[str] = None

class PredictRequest(BaseModel):
    schedule_path: str
    data_dir: str
    models_dir: str
    output_path: Optional[str] = None
    prop_lines: Optional[Dict[str, float]] = None

# file: src/nfl_prop_predictor/features.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from .utils import TARGETS, add_days_rest, ensure_columns, normalize_position, normalize_team, parse_date

ROLLS: List[int] = [3,5,10]
CONTEXT_COLS: List[str] = ["home_away","spread","total","snap_pct","targets","rush_att","pass_att","team_points","opp_points"]

def _numericify_cols(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def load_games(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_columns(df, ["player_id","player_name","team","position","opp_team","date","season","week","home_away"], "games.csv")
    df = parse_date(df, "date")
    df["team"] = df["team"].map(normalize_team)
    df["opp_team"] = df["opp_team"].map(normalize_team)
    df["position"] = df["position"].map(normalize_position)
    for t in TARGETS:
        df[t] = pd.to_numeric(df[t], errors="coerce") if t in df.columns else np.nan
    _numericify_cols(df, CONTEXT_COLS)
    return df

def load_defenses(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_columns(df, ["season","team","position","metric","allowed_per_game","rank"], "defenses.csv")
    df["team"] = df["team"].map(normalize_team)
    return df

def load_schedule(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_columns(df, ["player_id","player_name","team","position","opp_team","date","season","week","home_away"], "schedule.csv")
    df = parse_date(df, "date")
    df["team"] = df["team"].map(normalize_team)
    df["opp_team"] = df["opp_team"].map(normalize_team)
    df["position"] = df["position"].map(normalize_position)
    _numericify_cols(df, CONTEXT_COLS + [c for c in df.columns if c.startswith("line_")])
    return df

def make_rolling_features(g: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    g = g.sort_values("date")
    for t in target_cols:
        g[f"{t}_lag1"] = g[t].shift(1)
        for n in ROLLS:
            roll = g[t].rolling(n, min_periods=1)
            g[f"{t}_roll{n}_mean"] = roll.mean().shift(1)
            g[f"{t}_roll{n}_std"] = roll.std().shift(1)
    return g

def join_defense(game_df: pd.DataFrame, def_df: pd.DataFrame) -> pd.DataFrame:
    piv = def_df.pivot_table(index=["season","team","position"], columns="metric", values="allowed_per_game", aggfunc="mean").reset_index()
    piv.columns = ["season","team","position", *[f"opp_def_{c}_allowed" for c in piv.columns if c not in {"season","team","position"}]]
    out = game_df.merge(
        piv,
        left_on=["season","opp_team","position"],
        right_on=["season","team","position"],
        how="left",
        suffixes=("","_drop")
    )
    drop_cols = [c for c in out.columns if c.endswith("_drop")]
    return out.drop(columns=drop_cols)

def build_training_frame(games: pd.DataFrame, defenses: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()
    df = add_days_rest(df)
    df = df.sort_values(["player_id","date"])
    df = df.groupby("player_id", group_keys=False).apply(lambda g: make_rolling_features(g, TARGETS))
    df = join_defense(df, defenses)
    df["is_home"] = df["home_away"].astype(str).str.upper().eq("H").astype(int)
    _numericify_cols(df, CONTEXT_COLS)
    return df

def feature_targets(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    feat_cols: List[str] = [
        "is_home","days_rest","season","week","spread","total",
        f"{target}_lag1",
        f"{target}_roll3_mean", f"{target}_roll5_mean", f"{target}_roll10_mean",
        f"{target}_roll3_std", f"{target}_roll5_std", f"{target}_roll10_std",
    ]
    opp_col = f"opp_def_{target}_allowed"
    if opp_col in df.columns: feat_cols.append(opp_col)
    for c in ["snap_pct","targets","rush_att","pass_att"]:
        if c in df.columns: feat_cols.append(c)
    X = df.reindex(columns=feat_cols).fillna(0.0)
    y = df[target].astype(float)
    return X, y

# file: src/nfl_prop_predictor/train.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
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
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - val_size)); split_idx = max(1, min(split_idx, len(df) - 1))
    return df.iloc[:split_idx], df.iloc[split_idx:]

def train_models(data_dir: str, models_dir: str, config: TrainConfig = TrainConfig()) -> pd.DataFrame:
    data_dir_path = Path(data_dir); models_dir_path = Path(models_dir); models_dir_path.mkdir(parents=True, exist_ok=True)
    games = load_games(str(data_dir_path / "games.csv"))
    defenses = load_defenses(str(data_dir_path / "defenses.csv"))
    df = build_training_frame(games, defenses)
    results: List[tuple] = []
    for target in TARGETS:
        rows = df[df.get(f"{target}_lag1").notna()].copy()
        if rows.empty or len(rows) < 20: continue
        train_df, val_df = _time_aware_split(rows, config.validation_size)
        X_tr, y_tr = feature_targets(train_df, target)
        X_va, y_va = feature_targets(val_df, target)
        model = HistGradientBoostingRegressor(random_state=config.random_state)
        model.fit(X_tr, y_tr)
        va_pred = model.predict(X_va)
        rmse = float(mean_squared_error(y_va, va_pred, squared=False)); r2 = float(r2_score(y_va, va_pred))
        resid_std = float(np.std(y_va - va_pred, ddof=1)) if len(y_va) > 1 else 10.0  # why: for P(over)
        try:
            pi = permutation_importance(model, X_va, y_va, n_repeats=5, random_state=config.random_state)
            top_feats = ", ".join(f"{n}:{m:.3f}" for n, m in sorted(zip(X_va.columns, pi.importances_mean), key=lambda x: -x[1])[:5])
        except Exception:
            top_feats = "n/a"
        out_dir = models_dir_path / target; out_dir.mkdir(parents=True, exist_ok=True)
        dump(model, out_dir / "model.joblib")
        (out_dir / "features.txt").write_text("\n".join(X_tr.columns))
        (out_dir / "model_card.json").write_text(
            ModelCard(target=target, features=list(X_tr.columns), residual_std=resid_std,
                      train_rows=int(len(X_tr)), val_rows=int(len(X_va)),
                      notes=f"RMSE={rmse:.2f}, R2={r2:.3f}, top_feats={top_feats}").model_dump_json(indent=2)
        )
        results.append((target, rmse, r2, resid_std))
    summary = pd.DataFrame(results, columns=["target","rmse","r2","resid_std"]) if results else pd.DataFrame(columns=["target","rmse","r2","resid_std"])
    summary.to_csv(models_dir_path / "training_summary.csv", index=False)
    return summary

# file: src/nfl_prop_predictor/predict.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import norm
from .features import build_training_frame, load_defenses, load_games, load_schedule
from .utils import TARGETS

def _load_model_bundle(models_dir: Path, target: str):
    mdir = models_dir / target
    if not mdir.exists(): raise FileNotFoundError(f"Missing model dir: {mdir}")
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
            preds[f"prob_over_{target}"] = (1 - norm.cdf(z)).clip(0, 1)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True); preds.to_csv(output_path, index=False)
    return preds

# file: src/nfl_prop_predictor/__init__.py
from .train import train_models
__all__ = ["train_models"]

# file: src/nfl_prop_predictor/serve.py  (unchanged)
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pathlib import Path
from .predict import predict_schedule
from .data_models import PredictRequest

app = FastAPI(title="NFL Prop Predictor API", version="1.0")

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if not Path(req.schedule_path).exists():
            raise HTTPException(400, "schedule_path not found")
        df = predict_schedule(req.data_dir, req.models_dir, req.schedule_path, req.output_path)
        return {"rows": len(df), "columns": list(df.columns), "preview": df.head(20).to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
