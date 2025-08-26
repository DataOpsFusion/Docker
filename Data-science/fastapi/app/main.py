# app/main.py
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI()

# ----- Load default model on startup (models/apple.joblib) -----
@app.on_event("startup")
def _load_default_model():
    base = Path(__file__).resolve().parent.parent        # /.../stock
    model_path = base / "models" / "apple.joblib"        # /.../stock/models/apple.joblib
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    app.state.model = load(model_path)

# ----- Request body schema -----
class ModelParams(BaseModel):
    current_price: float
    change: float
    percent_change: float
    high: float
    low: float
    open: float
    previous_close: float

# ----- Health -----
@app.get("/status")
def status():
    return {"status": "ok"}

# ----- Predict (uses default loaded model) -----
@app.post("/predict")
def predict(params: ModelParams):
    X = preprocess_data(params)
    y = app.state.model.predict(X)
    if hasattr(y, "tolist"):
        y = y.tolist()
    # if it's a single value array, return a scalar
    if isinstance(y, list) and len(y) == 1:
        y = y[0]
    return {"prediction": y}

def preprocess_data(params: ModelParams):
    live = {
        "current_price": float(params.current_price),
        "change": float(params.change),
        "percent_change": float(params.percent_change),
        "high": float(params.high),
        "low": float(params.low),
        "open": float(params.open),
        "previous_close": float(params.previous_close),
    }
    live["range"] = live["high"] - live["low"]
    live["intraday_return"] = live["current_price"] - live["open"]
    # feature order expected by your pipeline:
    return pd.DataFrame([{
        "close": live["current_price"],
        "range": live["range"],
        "intraday_return": live["intraday_return"],
        "percent_change": live["percent_change"],
    }])