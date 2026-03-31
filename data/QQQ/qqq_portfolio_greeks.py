import pandas as pd
from pathlib import Path
import argparse

CONTRACT_SIZE = 100


def compute_portfolio_greeks(input_csv, output_csv=None):
    input_path = Path(input_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path)

    required = {"delta", "gamma", "theta", "vega"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Portfolio Greeks
    df["portfolio_delta"] = df["delta"] * CONTRACT_SIZE
    df["portfolio_gamma"] = df["gamma"] * CONTRACT_SIZE
    df["portfolio_theta"] = df["theta"] * CONTRACT_SIZE
    df["portfolio_vega"] = df["vega"] * CONTRACT_SIZE

    if output_csv is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_portfolio.csv"
        )
    else:
        output_path = Path(output_csv)

    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    out = compute_portfolio_greeks(args.input_csv, args.output)
    print(f"Saved to: {out}")