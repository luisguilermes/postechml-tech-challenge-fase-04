import os
from pathlib import Path
import pandas as pd
import yfinance as yf


def fetch_symbol(symbol: str, start: str, end: str, interval: str = "1d") -> Path:
    """Download historical data for `symbol` and save to data/<symbol>.csv"""
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data downloaded for {symbol} {start}:{end}")
    out_path = out_dir / f"{symbol.replace('/','_')}.csv"
    df.to_csv(out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", default="1d")
    args = parser.parse_args()
    p = fetch_symbol(args.symbol, args.start, args.end, args.interval)
    print("Saved data to", p)
