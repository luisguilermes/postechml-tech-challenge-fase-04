"""Simple wrapper script to fetch a sample dataset using src.data_fetch"""

from src.data_fetch import fetch_symbol


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-07-20")
    parser.add_argument("--interval", default="1d")
    args = parser.parse_args()
    p = fetch_symbol(args.symbol, args.start, args.end, args.interval)
    print("Saved to", p)


if __name__ == "__main__":
    main()
