# file: src/nfl_prop_predictor/utils.py
from __future__ import annotations

from typing import List, Optional

import pandas as pd

# Common franchise aliases → current abbreviations
TEAM_ALIASES = {
    "JAX": "JAC",
    "WSH": "WAS",
    "LA": "LAR",
}


def normalize_team(team: Optional[str]) -> Optional[str]:
    """Uppercase + alias map; returns None for missing."""
    if team is None or (isinstance(team, float) and pd.isna(team)):
        return None
    t = str(team).strip().upper()
    return TEAM_ALIASES.get(t, t)


def normalize_position(pos: Optional[str]) -> Optional[str]:
    """Uppercase; constrain to common skill positions if possible."""
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return None
    p = str(pos).strip().upper()
    return p if p in {"QB", "RB", "WR", "TE"} else p


def parse_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """
    Parse a column as datetime and keep only the calendar date.
    (Avoids tz headaches by dropping timezone info.)
    """
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def ensure_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    """Raise with a clear message if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days between this game and previous game per player."""
    out = df.sort_values(["player_id", "date"]).copy()
    prev = out.groupby("player_id")["date"].shift(1)
    d0 = pd.to_datetime(out["date"], errors="coerce")
    d1 = pd.to_datetime(prev, errors="coerce")
    days = (d0 - d1).dt.days
    out["days_rest"] = days.fillna(7).clip(lower=0, upper=30)
    return out


# Targets supported by the project
TARGETS: List[str] = [
    "pass_yards",
    "rush_yards",
    "rec_yards",
    "receptions",
    "pass_td",
    "rush_td",
    "rec_td",
]
