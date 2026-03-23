import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def nearest_strike(spot, step=1):
    return round(spot / step) * step


def build_synthetic_options(input_csv: str, output_csv: str | None = None) -> Path:
    base_dir = Path(__file__).parent
    input_path = base_dir / input_csv

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")

    df = pd.read_csv(input_path)

    # --- CLEAN COLUMN NAMES ---
    df.columns = df.columns.str.strip()
    col_map = {col.lower(): col for col in df.columns}

    required = {"date", "close", "realized_vol_20d"}
    if not required.issubset(col_map):
        raise ValueError("CSV must contain Date, Close, realized_vol_20d")

    date_col = col_map["date"]
    close_col = col_map["close"]
    vol_col = col_map["realized_vol_20d"]

    # --- PARSE ---
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")

    df = df.dropna(subset=[date_col, close_col, vol_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # --- CONSTANTS ---
    r = 0.03
    dte = 30
    T = dte / 365

    # ✅ ADD REQUIRED COLUMNS
    df["spot"] = df[close_col]                          # spot = close
    df["strike"] = df["spot"].apply(nearest_strike)     # ATM strike
    df["dte"] = dte                                     # days to expiry
    df["T"] = T                                         # time in years
    df["sigma"] = df[vol_col]                           # realized vol
    df["r"] = r                                         # risk-free rate
    df["option_type"] = "call"                          # option type


    # --- OUTPUT ---
    if output_csv is None:
        output_path = input_path.with_name(f"{input_path.stem}_with_options.csv")
    else:
        output_path = Path(output_csv)

    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add synthetic ATM option columns to dataset"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="IWM_data_features.csv",
        help="Input CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output file",
    )

    args = parser.parse_args()

    output_file = build_synthetic_options(args.input_csv, args.output)
    print(f"Saved output to: {output_file}")