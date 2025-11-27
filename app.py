# file: app.py
"""
Streamlit UI for NFL Prop Predictor.
Run with:
    PYTHONPATH=src streamlit run app.py
Requires:
    pip install streamlit
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

# Local package imports (expects project layout with src/)
with st.sidebar:
    st.markdown("### Paths")
data_dir = st.sidebar.text_input("Data directory", "./data")
models_dir = st.sidebar.text_input("Models directory", "./models")

# Lazy import project modules after PYTHONPATH is set by the user on run
try:
    from nfl_prop_predictor.train import train_models
    from nfl_prop_predictor.predict import predict_schedule
    from nfl_prop_predictor.utils import TARGETS
except Exception as e:  # why: give users actionable hint if PYTHONPATH missing
    st.error(
        f"Could not import project modules: {e}\n\n"
        "Run with `PYTHONPATH=src streamlit run app.py` or install the package."
    )
    TARGETS = ["pass_yards","rush_yards","rec_yards","receptions","pass_td","rush_td","rec_td"]  # fallback for UI

st.set_page_config(page_title="NFL Prop Predictor", layout="wide")

st.title("🏈 NFL Prop Predictor (ML + Opponent Defense)")

@st.cache_data(show_spinner=False)
def _read_csv_bytes(uploaded) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(uploaded.getvalue()))

def _write_temp_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name

def _predict_from_df(schedule_df: pd.DataFrame, data_dir: str, models_dir: str) -> pd.DataFrame:
    sched_path = _write_temp_csv(schedule_df)
    try:
        preds = predict_schedule(data_dir=data_dir, models_dir=models_dir, schedule_path=sched_path, output_path=None)
    finally:
        try:
            os.unlink(sched_path)
        except OSError:
            pass
    return preds

with st.sidebar:
    st.markdown("### Schedule")
    sched_upload = st.file_uploader("Upload schedule.csv", type=["csv"])
    global_target = st.selectbox("Prop target (optional)", ["(none)"] + TARGETS, index=0)
    global_line = st.number_input("Global prop line (applies to selected target)", value=0.0, step=0.5)
    st.caption("If your schedule already has `line_*` columns, this global line is optional.")
    st.divider()
    col_a, col_b = st.columns(2)
    do_train = col_a.button("🚀 Train Models")
    do_predict = col_b.button("🔮 Predict")

# TRAIN
if do_train:
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
preds_df: Optional[pd.DataFrame] = None
if do_predict:
    if sched_upload is None:
        st.warning("Upload a schedule CSV to predict.")
    else:
        try:
            sched_df = _read_csv_bytes(sched_upload)
            with st.spinner("Building features and generating predictions..."):
                preds_df = _predict_from_df(sched_df, data_dir=data_dir, models_dir=models_dir)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# UI when we have predictions
if preds_df is not None:
    st.subheader("Predictions")
    # Optional global prop line → compute P(over) for selected target
    if global_target != "(none)":
        pred_col = f"pred_{global_target}"
        sigma_col = f"unc_sigma_{global_target}"
        if pred_col in preds_df.columns and sigma_col in preds_df.columns:
            z = (float(global_line) - preds_df[pred_col].astype(float)) / preds_df[sigma_col].replace(0, 1.0)
            preds_df[f"prob_over_{global_target}_global"] = (1 - norm.cdf(z)).clip(0, 1)
        else:
            st.info(f"No predictions found for target `{global_target}`. Train models first or check data.")

    # Basic filters
    filt_cols = st.columns(4)
    name_filter = filt_cols[0].text_input("Filter: player_name contains")
    team_filter = filt_cols[1].text_input("Filter: team (e.g., KC)")
    opp_filter = filt_cols[2].text_input("Filter: opp_team (e.g., BUF)")
    pos_filter = filt_cols[3].text_input("Filter: position (QB/RB/WR/TE)")

    df_view = preds_df.copy()
    if name_filter:
        df_view = df_view[df_view["player_name"].astype(str).str.contains(name_filter, case=False, na=False)]
    if team_filter:
        df_view = df_view[df_view["team"].astype(str).str.upper().eq(team_filter.strip().upper())]
    if opp_filter:
        df_view = df_view[df_view["opp_team"].astype(str).str.upper().eq(opp_filter.strip().upper())]
    if pos_filter:
        df_view = df_view[df_view["position"].astype(str).str.upper().eq(pos_filter.strip().upper())]

    st.dataframe(df_view, use_container_width=True, hide_index=True)

    # Download
    st.download_button(
        "Download predictions.csv",
        preds_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

    # Quick single-player viz
    st.subheader("Single Player Quick Look")
    left, right = st.columns([2, 1])
    with right:
        player_opts = preds_df["player_name"].dropna().unique().tolist()
        player_sel = st.selectbox("Player", player_opts if player_opts else ["(none)"])
        target_sel = st.selectbox("Target", TARGETS)
        line_val = st.number_input("Chart line", value=float(global_line if global_target == target_sel else 0.0), step=0.5)

    if player_sel != "(none)":
        # Assemble a small frame with predicted value and optional line
        row = preds_df[preds_df["player_name"] == player_sel].head(1)
        pred_col = f"pred_{target_sel}"
        if not row.empty and pred_col in preds_df.columns:
            import matplotlib.pyplot as plt

            yhat = float(row.iloc[0][pred_col])
            fig, ax = plt.subplots()
            ax.bar(["prediction"], [yhat])
            ax.set_ylabel(target_sel.replace("_", " ").title())
            ax.set_title(f"{player_sel} — Predicted {target_sel}")
            if line_val and line_val > 0:
                ax.axhline(line_val)  # line only
            st.pyplot(fig)
        else:
            st.info("No prediction available for selected target/player.")

# Footer
st.caption(
    "Tip: Put your CSVs in the data folder (games.csv, defenses.csv, schedule.csv). "
    "Train once, then predict with any future schedule."
)
