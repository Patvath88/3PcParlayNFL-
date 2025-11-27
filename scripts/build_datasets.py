# file: scripts/build_datasets.py
"""
Builds the 3 CSVs required by the NFL Prop Predictor:
  - data/games.csv
  - data/defenses.csv
  - data/schedule.csv

Usage:
  pip install nflreadpy pandas
  python scripts/build_datasets.py --seasons 2019-2025 --out ./data

Notes:
  - games.csv: one row per player-game with core targets + context fields.
  - defenses.csv: per season/team/position/metric "allowed_per_game".
  - schedule.csv: starter file for the next upcoming NFL week (no lines).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Pull data from nflverse
import nflreadpy as nfl  # https://github.com/nflverse/nflreadpy

TARGETS = ["pass_yards", "rush_yards", "rec_yards", "receptions", "pass_td", "rush_td", "rec_td"]


def _parse_seasons(spec: str) -> List[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def _position_from_nflverse(row: pd.Series) -> str:
    # Prefer the position coded in player stats; fallback to roster position if present
    pos = str(row.get("position", "") or row.get("player_position", "") or "").upper()
    return pos if pos in {"QB", "RB", "WR", "TE"} else pos


def build_games(seasons: List[int]) -> pd.DataFrame:
    ps = nfl.load_player_stats(seasons)  # Polars DataFrame
    ps = ps.to_pandas()

    # Expect these columns to exist; map to our schema
    # nflverse columns are fairly stable but may evolve.
    mapping = {
        "player_id": "player_id",
        "player_name": "player_name",
        "recent_team": "team",
        "opponent_team": "opp_team",
        "game_date": "date",
        "season": "season",
        "week": "week",
        "home_away": "home_away",
        # targets (common names below; adjust if nflverse renames)
        "passing_yards": "pass_yards",
        "rushing_yards": "rush_yards",
        "receiving_yards": "rec_yards",
        "receptions": "receptions",
        "passing_tds": "pass_td",
        "rushing_tds": "rush_td",
        "receiving_tds": "rec_td",
        # Optional context if available
        "team_score": "team_points",
        "opponent_score": "opp_points",
        "spread_line": "spread",
        "total_line": "total",
        "targets": "targets",
        "rushing_attempts": "rush_att",
        "passing_attempts": "pass_att",
        "offense_snapshare": "snap_pct",
    }

    missing = [src for src in mapping if src not in ps.columns]
    # We tolerate missing context fields; targets should exist.
    # Filter mapping to existing columns:
    mapping = {src: dst for src, dst in mapping.items() if src in ps.columns}
    df = ps.rename(columns=mapping)

    # Minimal required fields
    req = ["player_id", "player_name", "team", "opp_team", "date", "season", "week", "home_away"]
    for col in req:
        if col not in df.columns:
            raise RuntimeError(f"games.csv missing required column from source: {col}")

    # Position
    if "position" not in df.columns:
        df["position"] = df.apply(_position_from_nflverse, axis=1)

    # Casts
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["team"] = df["team"].str.upper()
    df["opp_team"] = df["opp_team"].str.upper()
    df["home_away"] = df["home_away"].fillna("").str.upper().map({"HOME": "H", "AWAY": "A", "H": "H", "A": "A"})
    for t in TARGETS:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce")
        else:
            df[t] = np.nan

    # Optional numerics
    for c in ["team_points", "opp_points", "spread", "total", "snap_pct", "targets", "rush_att", "pass_att"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only columns we care about
    keep = list({*req, "position", *TARGETS, "team_points", "opp_points", "spread", "total", "snap_pct", "targets", "rush_att", "pass_att"})
    return df[keep].sort_values(["player_id", "date"])


def build_defense_allowed(games: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates allowed stats per opponent team × position × season.
    Output schema: season, team, position, metric, allowed_per_game, rank
    """
    long_rows = []
    for metric in TARGETS:
        if metric not in games.columns:
            continue
        tmp = (
            games.groupby(["season", "opp_team", "position"], as_index=False)[metric]
            .sum()
            .rename(columns={"opp_team": "team", metric: "sum_val"})
        )
        # Count games played against that defense × position
        counts = games.groupby(["season", "opp_team", "position"], as_index=False).size().rename(columns={"opp_team": "team", "size": "n"})
        tmp = tmp.merge(counts, on=["season", "team", "position"], how="left")
        tmp["allowed_per_game"] = tmp["sum_val"] / tmp["n"].replace(0, np.nan)
        tmp["metric"] = metric
        long_rows.append(tmp[["season", "team", "position", "metric", "allowed_per_game"]])

    out = pd.concat(long_rows, ignore_index=True)
    # Rank: lower allowed_per_game = "better" defense rank 1, so rank ascending
    out["rank"] = out.groupby(["season", "position", "metric"])["allowed_per_game"].rank(method="min", ascending=True)
    out["rank"] = out["rank"].astype(int)
    return out.sort_values(["season", "position", "metric", "rank"])


def build_schedule_next_week(seasons: List[int]) -> pd.DataFrame:
    """
    Creates a barebones schedule.csv for the next upcoming NFL week (teams only).
    You typically replace/expand this with the specific players you care about.
    """
    sched = nfl.load_schedules(max(seasons))  # latest season schedule
    sched = sched.to_pandas()
    # Identify next upcoming week
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    upcoming = sched[sched["gameday"] >= pd.Timestamp("today").normalize()].sort_values("gameday")
    if upcoming.empty:
        # fallback: last completed week + 1
        cur_week = int(sched["week"].max())
        next_wk = cur_week + 1
    else:
        next_wk = int(upcoming["week"].iloc[0])

    wk = sched[sched["week"] == next_wk].copy()
    wk["date"] = wk["gameday"].dt.date
    # Build team-level rows as placeholders (user should expand with players)
    home = wk[["home_team", "away_team", "season", "week", "date"]].rename(columns={"home_team": "team", "away_team": "opp_team"})
    home["home_away"] = "H"
    away = wk[["away_team", "home_team", "season", "week", "date"]].rename(columns={"away_team": "team", "home_team": "opp_team"})
    away["home_away"] = "A"
    teams = pd.concat([home, away], ignore_index=True).drop_duplicates()

    # Minimal schema with placeholders for player info (fill later)
    teams["player_id"] = ""
    teams["player_name"] = ""
    teams["position"] = ""
    # Optional odds lines can be added later as columns line_<target>
    cols = ["player_id", "player_name", "team", "position", "opp_team", "date", "season", "week", "home_away"]
    return teams[cols].sort_values(["date", "team"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True, help="e.g. 2019-2025 or 2021,2022,2023")
    ap.add_argument("--out", default="./data")
    args = ap.parse_args()

    seasons = _parse_seasons(args.seasons)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading player stats for seasons: {seasons} ...")
    games = build_games(seasons)
    games.to_csv(out_dir / "games.csv", index=False)
    print(f"Wrote {out_dir/'games.csv'} with {len(games):,} rows")

    print("Computing defense allowed metrics ...")
    defenses = build_defense_allowed(games)
    defenses.to_csv(out_dir / "defenses.csv", index=False)
    print(f"Wrote {out_dir/'defenses.csv'} with {len(defenses):,} rows")

    print("Building starter schedule for the next week ...")
    schedule = build_schedule_next_week(seasons)
    schedule.to_csv(out_dir / "schedule.csv", index=False)
    print(f"Wrote {out_dir/'schedule.csv'} with {len(schedule):,} rows")

    print("Done. You can now run training with these CSVs.")


if __name__ == "__main__":
    main()
