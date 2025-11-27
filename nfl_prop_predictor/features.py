from future import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple
from .utils import normalize_team, normalize_position, parse_date, ensure_columns, add_days_rest, TARGETS

ROLLS = [3, 5, 10]
CONTEXT_COLS = ["home_away","spread","total","snap_pct","targets","rush_att","pass_att","team_points","opp_points"]

def load_games(path: str) -> pd.DataFrame:
df = pd.read_csv(path)
ensure_columns(df, ["player_id","player_name","team","position","opp_team","date","season","week","home_away"], "games.csv")
df = parse_date(df, "date")
df["team"] = df["team"].map(normalize_team)
df["opp_team"] = df["opp_team"].map(normalize_team)
df["position"] = df["position"].map(normalize_position)
for t in TARGETS:
if t in df.columns:
df[t] = pd.to_numeric(df[t], errors="coerce")
else:
df[t] = np.nan
for c in CONTEXT_COLS:
if c in df.columns:
df[c] = pd.to_numeric(df[c], errors="coerce")
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
for c in CONTEXT_COLS:
if c in df.columns:
df[c] = pd.to_numeric(df[c], errors="coerce")
return df

def make_rolling_features(g: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
g = g.sort_values("date")
for t in target_cols:
g[f"{t}_lag1"] = g[t].shift(1)
for n in ROLLS:
g[f"{t}_roll{n}_mean"] = g[t].rolling(n, min_periods=1).mean().shift(1)
g[f"{t}_roll{n}_std"] = g[t].rolling(n, min_periods=1).std().shift(1)
return g

def join_defense(game_df: pd.DataFrame, def_df: pd.DataFrame) -> pd.DataFrame:
piv = def_df.pivot_table(index=["season","team","position"], columns="metric", values="allowed_per_game")
piv.columns = [f"opp_def_{c}_allowed" for c in piv.columns]
piv = piv.reset_index()
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
df = df.sort_values(["player_id", "date"])
df = df.groupby("player_id", group_keys=False).apply(lambda g: make_rolling_features(g, TARGETS))
df = join_defense(df, defenses)
df["is_home"] = (df["home_away"].astype(str).str.upper().eq("H")).astype(int)
for c in CONTEXT_COLS:
if c in df.columns:
df[c] = pd.to_numeric(df[c], errors="coerce")
return df

def feature_targets(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
feat_cols = [
"is_home","days_rest","season","week",
"spread","total",
f"{target}_lag1",
f"{target}_roll3_mean", f"{target}_roll5_mean", f"{target}_roll10_mean",
f"{target}_roll3_std", f"{target}_roll5_std", f"{target}roll10_std",
]
opp_col = f"opp_def{target}_allowed"
if opp_col in df.columns:
feat_cols.append(opp_col)
for c in ["snap_pct","targets","rush_att","pass_att"]:
if c in df.columns:
feat_cols.append(c)
X = df.reindex(columns=feat_cols).copy()
y = df[target].astype(float)
return X, y
