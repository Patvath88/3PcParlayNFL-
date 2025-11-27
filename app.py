# file: app.py  (root)
"""
Streamlit UI for NFL Prop Predictor with free data fetch (no API key) via nflreadpy.
Run:
  PYTHONPATH=src streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
from math import erf, sqrt
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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

# --- Free data source: nflverse via nflreadpy (no API key) ---
import nflreadpy as nfl  # public CC-BY datasets

# ---------- Helpers ----------
def normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    # Standard normal CDF without SciPy.
    return 0.5 * (1.0 + np.vectorize(lambda v: erf(v / sqrt(2.0)))(x))

def _pick_player_id(df: pd.DataFrame) -> str:
    for c in ["player_id", "gsis_id", "player_gsis_id", "nfl_id", "pfr_id", "esb_id"]:
        if c in df.columns:
            return c
    raise RuntimeError("No player id column found in source data.")

def _position_from_row(row: pd.Series) -> str:
    pos = str(row.get("position") or row.get("player_position") or "").upper()
    return pos if pos in {"QB","RB","WR","TE"} else pos

def _parse_seasons(spec: str) -> List[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]

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
        "passing_yards": "pass_yards",
        "rushing_yards": "rush_yards",
        "receiving_yards": "rec_yards",
        "receptions": "receptions",
        "passing_tds": "pass_td",
        "rushing_tds": "rush_td",
        "receiving_tds": "rec_td",
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
        df[t] = pd.to_numeric(df[t], errors="coerce") if t in df.columns else np.nan
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

    stats = nfl.load_player_stats(season_latest).to_pandas()
    pid = _pick_player_id(stats)
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

    def team_rows(frame, team_col, opp_col, hoa):
        df = frame[[team_col, opp_col, "season", "week", "date"]].rename(columns={team_col:"team", opp_col:"opp_team"}).copy()
        df["home_away"] = hoa
        return df

    games_teams = pd.concat(
        [team_rows(wk, "home_team", "away_team", "H"), team_rows(wk, "away_team", "home_team", "A")],
        ignore_index=True,
    ).drop_duplicates()

    sched_players = players.merge(games_teams, on="team", how="inner")
    sched_players = sched_players[["player_id","player_name","team","position","opp_team","date","season","week","home_away"]].drop_duplicates()
    return sched_players.sort_values(["team","position","player_name"])

@st.cache_data(show_spinner=True)
def fetch_and_build(out_dir: str, seasons_text: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seasons = _parse_seasons(seasons_text)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    games = _build_games(seasons)
    defenses = _build_defenses(games)
    schedule = _build_player_schedule_next_week(seasons)
    games.to_csv(out / "games.csv", index=False)
    defenses.to_csv(out / "defenses.csv", index=False)
    schedule.to_csv(out / "schedule.csv", index=False)
    return games, defenses, schedule

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
    seasons_text = st.text_input("Seasons (e.g., 2022-2025 or 2023,2024,2025)", "2022-2025")
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
                st.subheader("Predictions")
                gcol1, gcol2 = st.columns(2)
                tgt = gcol1.selectbox("Prop target", TARGETS, index=2)
                line = gcol2.number_input("Global line (optional)", value=0.0, step=0.5)
                if line > 0:
                    pred_col = f"pred_{tgt}"
                    sigma_col = f"unc_sigma_{tgt}"
                    if pred_col in preds.columns and sigma_col in preds.columns:
                        z = (float(line) - preds[pred_col].astype(float)) / preds[sigma_col].replace(0, 1.0)
                        preds[f"prob_over_{tgt}_global"] = (1 - normal_cdf(z)).clip(0, 1)
                st.dataframe(preds, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download predictions.csv",
                    preds.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.caption("Tip: Re-run Fetch with a different season range to refresh data.")
