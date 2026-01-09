import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

from src.models.preprocess import prepare_data


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def mse_to_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    return mae, rmse, mape


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = Path(args.csv)
    window = args.window
    X_train, y_train, X_val, y_val, scaler = prepare_data(
        csv_path, window=window, test_size=args.test_size
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTMModel(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                val_preds.append(out.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_trues = np.vstack(val_trues)
        mae, rmse, mape = mse_to_metrics(val_trues, val_preds)

        print(
            f"Epoch {epoch}/{args.epochs} - train_loss={np.mean(train_losses):.6f} val_mae={mae:.6f} rmse={rmse:.6f} mape={mape:.3f}%"
        )

    # Save model and scaler
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"lstm_{csv_path.stem}.pt"
    scaler_path = out_dir / f"scaler_{csv_path.stem}.pkl"
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print("Saved model to", model_path)
    print("Saved scaler to", scaler_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/BBAS3.SA.csv")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    train(args)
