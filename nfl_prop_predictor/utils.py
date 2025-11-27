from future import annotations
import pandas as pd
from typing import List

TEAM_ALIASES = {"JAX": "JAC", "WSH": "WAS", "LA": "LAR"}

def normalize_team(team: str) -> str:
if pd.isna(team):
return team
t = str(team).strip().upper()
return TEAM_ALIASES.get(t, t)

def normalize_position(pos: str) -> str:
if pd.isna(pos):
return pos
p = str(pos).strip().upper()
return p if p in {"QB", "RB", "WR", "TE"} else p

def parse_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert("UTC").dt.date
return df

def ensure_columns(df: pd.DataFrame, cols: List[str], name: str):
missing = [c for c in cols if c not in df.columns]
if missing:
raise ValueError(f"{name} missing columns: {missing}")

def add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
df = df.sort_values(["player_id", "date"])
df["prev_date"] = df.groupby("player_id")["date"].shift(1)
df["days_rest"] = (pd.to_datetime(df["date"]) - pd.to_datetime(df["prev_date"])).dt.days
df["days_rest"] = df["days_rest"].fillna(7).clip(lower=0, upper=30)
return df.drop(columns=["prev_date"])

TARGETS = ["pass_yards","rush_yards","rec_yards","receptions","pass_td","rush_td","rec_td"]
