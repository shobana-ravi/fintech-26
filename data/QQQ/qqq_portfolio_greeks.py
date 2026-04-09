import pandas as pd
from pathlib import Path
import argparse

CONTRACT_SIZE = 100  # 1 option contract = 100 shares

def compute_portfolio_greeks(input_csv: str, output_csv: str | None = None) -> Path:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path)

    # --- CLEAN DUPLICATE COLUMNS ---
    df = df.loc[:, ~df.columns.duplicated()]

    required_cols = {"delta", "gamma", "theta", "vega"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- COMPUTE PORTFOLIO GREEKS ---
    df["portfolio_delta"] = df["delta"] * CONTRACT_SIZE
    df["portfolio_gamma"] = df["gamma"] * CONTRACT_SIZE
    df["portfolio_theta"] = df["theta"] * CONTRACT_SIZE
    df["portfolio_vega"] = df["vega"] * CONTRACT_SIZE

    if output_csv is None:
        output_path = input_path.with_name(f"{input_path.stem}_portfolio.csv")
    else:
        output_path = Path(output_csv)

    df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute portfolio Greeks from option Greeks")
    parser.add_argument("input_csv", help="CSV file with option Greeks")
    parser.add_argument("-o", "--output", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    output_file = compute_portfolio_greeks(args.input_csv, args.output)
    print(f"Saved output to: {output_file}")