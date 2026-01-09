#!/usr/bin/env python3
"""Compare model prediction (last point) with true last value.

Usage:
  python scripts/compare_last_prediction.py --csv data/BBAS3.SA.csv

The script takes the last `window` values excluding the final point,
calls the `/predict` API and prints prediction vs true value and errors.
"""
import argparse
import json
import sys
from pathlib import Path

import requests
import pandas as pd


def load_closes(csv_path: Path, date_format: str | None = None, date_col: int = 0):
    df = pd.read_csv(csv_path)
    col0 = df.columns[date_col]
    if date_format:
        df[col0] = pd.to_datetime(df[col0], format=date_format, errors="coerce")
    else:
        # let pandas infer dates (may warn if format ambiguous)
        df[col0] = pd.to_datetime(df[col0], errors="coerce")
    df = df.dropna(subset=[col0]).copy()
    df.set_index(col0, inplace=True)
    closes = df["Close"].astype(float).values
    return closes


def call_predict(url: str, symbol: str, history: list[float], timeout: int = 10):
    payload = {"symbol": symbol, "history": history}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main():
    p = argparse.ArgumentParser(description="Compare last prediction with true value")
    p.add_argument("--csv", default="data/BBAS3.SA.csv")
    p.add_argument("--url", default="http://127.0.0.1:8000/predict")
    p.add_argument("--symbol", default="BBAS3.SA")
    p.add_argument("--window", type=int, default=20)
    p.add_argument(
        "--date-format",
        default=None,
        help="optional date format to pass to pandas (eg. %%Y-%%m-%%d)",
    )
    p.add_argument("--timeout", type=int, default=10)
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("CSV not found:", csv_path, file=sys.stderr)
        sys.exit(2)

    closes = load_closes(csv_path, date_format=args.date_format)
    if len(closes) < args.window + 1:
        print("Not enough data: need at least", args.window + 1)
        sys.exit(3)

    history = closes[-(args.window + 1) : -1].tolist()
    true = float(closes[-1])

    try:
        resp = call_predict(args.url, args.symbol, history, timeout=args.timeout)
    except Exception as e:
        print("Request failed:", e, file=sys.stderr)
        sys.exit(4)

    pred = resp.get("prediction")
    try:
        pred = float(pred)
    except Exception:
        print("Could not parse prediction from response:", resp, file=sys.stderr)
        sys.exit(5)

    abs_err = abs(pred - true)
    mape = 100.0 * abs_err / true if true != 0 else None

    out = {
        "status_code": 200,
        "symbol": args.symbol,
        "true": true,
        "prediction": pred,
        "abs_err": abs_err,
        "mape_percent": mape,
        "raw_response": resp,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
