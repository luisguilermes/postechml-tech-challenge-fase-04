import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch

from src.models.preprocess import prepare_data


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


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error

    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def evaluate(
    csv_path: Path,
    model_path: Path,
    scaler_path: Path,
    window: int = 20,
    test_size: float = 0.2,
    batch_size: int = 128,
    out_dir: Path = Path("reports"),
):
    X_train, y_train, X_val, y_val, scaler = prepare_data(
        csv_path, window=window, test_size=test_size
    )

    device = torch.device("cpu")
    model = LSTMModel()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Prediction on validation set
    with torch.no_grad():
        inp = torch.tensor(X_val, dtype=torch.float32)
        preds = model(inp).cpu().numpy()

    # inverse scale
    preds_unscaled = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    y_val_unscaled = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()

    metrics = compute_metrics(y_val_unscaled, preds_unscaled)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{csv_path.stem}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_val_unscaled, label="true")
    plt.plot(preds_unscaled, label="pred")
    plt.legend()
    plt.title(f"Predictions vs True - {csv_path.stem}")
    plot_path = out_dir / f"pred_vs_true_{csv_path.stem}.png"
    plt.savefig(plot_path)
    plt.close()

    print("Saved metrics to", metrics_path)
    print("Saved plot to", plot_path)
    print("Metrics:", metrics)
    return metrics, metrics_path, plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/BBAS3.SA.csv")
    parser.add_argument("--model", default="models/lstm_BBAS3.SA.pt")
    parser.add_argument("--scaler", default="models/scaler_BBAS3.SA.pkl")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    evaluate(
        Path(args.csv),
        Path(args.model),
        Path(args.scaler),
        window=args.window,
        test_size=args.test_size,
    )
