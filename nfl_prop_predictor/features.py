# file: src/nfl_prop_predictor/features.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .utils import (
    TARGETS,
    add_days_rest,
    ensure_columns,
    normalize_position,
    normalize_team,
    parse_date,
)

# Rolling windows used for player history
ROLLS: List[int] = [3, 5, 10]

# Context columns that may exist in data; cast to numeric if present
CONTEXT_COLS: List[str] = [
    "home_away",
    "spread",
    "total",
    "snap_pct",
    "targets",
    "rush_att",
    "pass_att",
    "team_points",
    "opp_points",
]


def _numericify_cols(df: pd.DataFrame, cols: List[str]) -> None:
    """In-place numeric conversion for optional columns; non-numeric → NaN."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def load_games(path: str) -> pd.DataFrame:
    """Read historical player game logs and normalize."""
    df = pd.read_csv(path)
    ensure_columns(
        df,
        [
            "player_id",
            "player_name",
            "team",
            "position",
            "opp_team",
            "date",
            "season",
            "week",
            "home_away",
        ],
        "games.csv",
    )
    df = parse_date(df, "date")
    df["team"] = df["team"].map(normalize_team)
    df["opp_team"] = df["opp_team"].map(normalize_team)
    df["position"] = df["position"].map(normalize_position)

    # Targets → float; create missing targets as NaN so downstream code is stable
    for t in TARGETS:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce")
        else:
            df[t] = np.nan

    _numericify_cols(df, CONTEXT_COLS)
    return df


def load_defenses(path: str) -> pd.DataFrame:
    """Read opponent defensive ratings (allowed per game)."""
    df = pd.read_csv(path)
    ensure_columns(df, ["season", "team", "position", "metric", "allowed_per_game", "rank"], "defenses.csv")
    df["team"] = df["team"].map(normalize_team)
    # sanitize metric names to match TARGETS where applicable
    df["metric"] = df["metric"].astype(str)
    _numericify_cols(df, ["allowed_per_game", "rank"])
    return df


def load_schedule(path: str) -> pd.DataFrame:
    """Read upcoming schedule to predict."""
    df = pd.read_csv(path)
    ensure_columns(
        df,
        [
            "player_id",
            "player_name",
            "team",
            "position",
            "opp_team",
            "date",
            "season",
            "week",
            "home_away",
        ],
        "schedule.csv",
    )
    df = parse_date(df, "date")
    df["team"] = df["team"].map(normalize_team)
    df["opp_team"] = df["opp_team"].map(normalize_team)
    df["position"] = df["position"].map(normalize_position)
    _numericify_cols(df, CONTEXT_COLS)

    # If user provided prop lines, ensure numeric
    line_cols = [c for c in df.columns if c.startswith("line_")]
    _numericify_cols(df, line_cols)
    return df


def make_rolling_features(g: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """Per-player lags and rolling stats (shifted to avoid leakage)."""
    g = g.sort_values("date")
    for t in target_cols:
        g[f"{t}_lag1"] = g[t].shift(1)
        for n in ROLLS:
            roll = g[t].rolling(n, min_periods=1)
            g[f"{t}_roll{n}_mean"] = roll.mean().shift(1)
            g[f"{t}_roll{n}_std"] = roll.std().shift(1)
    return g


def join_defense(game_df: pd.DataFrame, def_df: pd.DataFrame) -> pd.DataFrame:
    """Wide-join defense metrics: each metric becomes `opp_def_<metric>_allowed`."""
    piv = (
        def_df.pivot_table(
            index=["season", "team", "position"],
            columns="metric",
            values="allowed_per_game",
            aggfunc="mean",
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )
    piv.columns = [
        "season",
        "team",
        "position",
        *[f"opp_def_{c}_allowed" for c in piv.columns if c not in {"season", "team", "position"}],
    ]
    out = game_df.merge(
        piv,
        left_on=["season", "opp_team", "position"],
        right_on=["season", "team", "position"],
        how="left",
        suffixes=("", "_drop"),
    )
    drop_cols = [c for c in out.columns if c.endswith("_drop")]
    return out.drop(columns=drop_cols)


def build_training_frame(games: pd.DataFrame, defenses: pd.DataFrame) -> pd.DataFrame:
    """Full feature table for training and for schedule featurization."""
    df = games.copy()
    df = add_days_rest(df)
    df = df.sort_values(["player_id", "date"])
    # why: groupby-apply preserves per-player time order for rolling features
    df = df.groupby("player_id", group_keys=False).apply(lambda g: make_rolling_features(g, TARGETS))
    df = join_defense(df, defenses)
    df["is_home"] = df["home_away"].astype(str).str.upper().eq("H").astype(int)
    _numericify_cols(df, CONTEXT_COLS)
    return df


def feature_targets(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target y for a given stat (e.g., 'rec_yards')."""
    feat_cols: List[str] = [
        "is_home",
        "days_rest",
        "season",
        "week",
        "spread",
        "total",
        f"{target}_lag1",
        f"{target}_roll3_mean",
        f"{target}_roll5_mean",
        f"{target}_roll10_mean",
        f"{target}_roll3_std",
        f"{target}_roll5_std",
        f"{target}_roll10_std",
    ]
    opp_col = f"opp_def_{target}_allowed"
    if opp_col in df.columns:
        feat_cols.append(opp_col)
    for c in ["snap_pct", "targets", "rush_att", "pass_att"]:
        if c in df.columns:
            feat_cols.append(c)

    # Align and fill missing to keep model input stable
    X = df.reindex(columns=feat_cols).fillna(0.0)
    y = df[target].astype(float)
    return X, y
