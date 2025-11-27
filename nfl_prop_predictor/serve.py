from future import annotations
from fastapi import FastAPI, HTTPException
from pathlib import Path
from .predict import predict_schedule
from .data_models import PredictRequest

app = FastAPI(title="NFL Prop Predictor API", version="1.0")

@app.post("/predict")
def predict(req: PredictRequest):
try:
if not Path(req.schedule_path).exists():
raise HTTPException(400, "schedule_path not found")
df = predict_schedule(req.data_dir, req.models_dir, req.schedule_path, req.output_path)
return {"rows": len(df), "columns": list(df.columns), "preview": df.head(20).to_dict(orient="records")}
except HTTPException:
raise
except Exception as e:
raise HTTPException(status_code=500, detail=str(e))
