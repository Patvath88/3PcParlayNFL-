from future import annotations
import typer
from .train import train_models
from .predict import predict_schedule, line_probability
from .data_models import TrainConfig

app = typer.Typer(add_completion=False, help="NFL Prop Predictor CLI")

@app.command()
def train(
data_dir: str = typer.Option(..., help="Folder containing games.csv, defenses.csv"),
models_dir: str = typer.Option(..., help="Output models dir"),
validation_size: float = 0.15,
min_games_for_features: int = 5,
random_state: int = 42,
):
cfg = TrainConfig(validation_size=validation_size, min_games_for_features=min_games_for_features, random_state=random_state)
summary = train_models(data_dir, models_dir, cfg)
typer.echo(summary.to_string(index=False))

@app.command()
def predict(
data_dir: str = typer.Option(...),
models_dir: str = typer.Option(...),
schedule: str = typer.Option(...),
out: str = typer.Option(None),
):
df = predict_schedule(data_dir, models_dir, schedule, out)
typer.echo(f"Wrote {len(df)} predictions" if out else df.head().to_string())

@app.command("line-prob")
def line_prob(
predictions: str = typer.Option(...),
out: str = typer.Option(None),
):
df = line_probability(predictions, out)
typer.echo(f"Wrote {len(df)} rows" if out else df.head().to_string())

if name == "main":
app()
