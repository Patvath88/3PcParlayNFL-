# NFL Prop Predictor (ML + Opponent Defense)

Train ML models on historical player game logs + opponent defensive ratings, then predict future props and `P(over)` for a given line.

## Data
`data/games.csv` (player-game logs)
- Required: player_id, player_name, team, position, opp_team, date (YYYY-MM-DD), season, week, home_away ("H"/"A")
- Targets: pass_yards, rush_yards, rec_yards, receptions, pass_td, rush_td, rec_td
- Optional: team_points, opp_points, spread, total, snap_pct, targets, rush_att, pass_att

`data/defenses.csv` (opponent allowed stats by team/position/metric)
- Required: season, team, position, metric, allowed_per_game, rank
- Example metrics = targets above

`data/schedule.csv` (future games)
- Required: player_id, player_name, team, position, opp_team, date, season, week, home_away
- Optional lines: line_pass_yards, line_rush_yards, line_rec_yards, line_receptions, line_pass_td, line_rush_td, line_rec_td
- Optional odds: spread, total
## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m nfl_prop_predictor.cli train --data-dir ./data --models-dir ./models
PYTHONPATH=src python -m nfl_prop_predictor.cli predict --data-dir ./data --models-dir ./models --schedule ./data/schedule.csv --out ./preds.csv
PYTHONPATH=src python -m nfl_prop_predictor.cli line-prob --predictions ./preds.csv --out ./props.csv
