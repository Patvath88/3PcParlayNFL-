# file: src/nfl_prop_predictor/data_models.py
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class TrainConfig(BaseModel):
    """Training hyperparameters."""
    validation_size: float = 0.15
    min_games_for_features: int = 5
    random_state: int = 42


class ModelCard(BaseModel):
    """Metadata persisted with each trained model."""
    target: str
    version: str = "1.0"
    features: List[str]
    residual_std: float
    train_rows: int
    val_rows: int
    notes: Optional[str] = None


class PredictRequest(BaseModel):
    """FastAPI request body for /predict."""
    schedule_path: str
    data_dir: str
    models_dir: str
    output_path: Optional[str] = None
    prop_lines: Optional[Dict[str, float]] = None
