from future import annotations
from pydantic import BaseModel
from typing import Optional, Dict, List

class TrainConfig(BaseModel):
validation_size: float = 0.15
min_games_for_features: int = 5
random_state: int = 42

class ModelCard(BaseModel):
target: str
version: str = "1.0"
features: List[str]
residual_std: float
train_rows: int
val_rows: int
notes: Optional[str] = None

class PredictRequest(BaseModel):
schedule_path: str
data_dir: str
models_dir: str
output_path: Optional[str] = None
prop_lines: Optional[Dict[str, float]] = None
