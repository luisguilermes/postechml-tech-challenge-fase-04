from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json

import torch
import joblib
import numpy as np

app = FastAPI(title="Stock LSTM API")


class PredictRequest(BaseModel):
    symbol: str
    history: Optional[List[float]] = None


# Simple LSTM model definition matching training script
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


_MODEL_CACHE = {}


def load_model_and_scaler(symbol: str, window: int = 20):
    key = f"{symbol}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    models_dir = Path("models")
    model_path = models_dir / f"lstm_{symbol}.pt"
    scaler_path = models_dir / f"scaler_{symbol}.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model or scaler not found for symbol: " + symbol)

    scaler = joblib.load(scaler_path)
    device = torch.device("cpu")
    model = LSTMModel()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    _MODEL_CACHE[key] = (model, scaler)
    return model, scaler


def predict_price(
    symbol: str, history: List[float], window: int = 20
) -> Optional[float]:
    """Predict next price given raw `history` list of close prices."""
    if history is None or len(history) < window:
        raise ValueError(f"Provide at least {window} historical close prices")

    model, scaler = load_model_and_scaler(symbol, window=window)

    arr = np.array(history[-window:], dtype=float).reshape(-1, 1)
    scaled = scaler.transform(arr)
    inp = torch.tensor(scaled.reshape(1, window, 1), dtype=torch.float32)
    with torch.no_grad():
        out = model(inp).cpu().numpy()
    pred_scaled = out.reshape(-1, 1)
    pred = scaler.inverse_transform(pred_scaled)[0, 0]
    return float(pred)


@app.get("/")
def root():
    return {"status": "ok", "info": "Stock LSTM API"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        pred = predict_price(req.symbol, req.history)
    except Exception as e:
        return {"error": str(e)}
    return {"symbol": req.symbol, "prediction": pred}
