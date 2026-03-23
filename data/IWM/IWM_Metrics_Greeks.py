# IWM_Metrics.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# --- BLACK-SCHOLES FUNCTIONS ---
def black_scholes_call(S, K, T, r, sigma):
    """Compute call option price using Black-Scholes formula."""
    if sigma <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def black_scholes_greeks(S, K, T, r, sigma):
    """Compute Greeks: delta, gamma, theta, vega for a call option."""
    if sigma <= 0 or T <= 0:
        return np.nan, np.nan, np.nan, np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, theta, vega

def nearest_strike(spot, step=1):
    return round(spot / step) * step

# --- MAIN FUNCTION ---
def build_synthetic_options(input_csv: str, output_csv: str | None = None) -> Path:
    base_dir = Path(__file__).parent
    input_path = base_dir / input_csv

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")

    df = pd.read_csv(input_path)
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
    df = df.dropna(subset=[date_col, close_col, vol_col]).sort_values(date_col).reset_index(drop=True)

    # --- CONSTANTS ---
    r = 0.03
    dte = 30
    T = dte / 365

    # --- SYNTHETIC OPTION COLUMNS ---
    df["spot"] = df[close_col]
    df["strike"] = df["spot"].apply(nearest_strike)
    df["dte"] = dte
    df["T"] = T
    df["sigma"] = df[vol_col]
    df["r"] = r
    df["option_type"] = "call"

    # --- COMPUTE PRICE & GREEKS ---
    results = df.apply(
        lambda row: black_scholes_greeks(
            S=row["spot"],
            K=row["strike"],
            T=row["T"],
            r=row["r"],
            sigma=row["sigma"]
        ),
        axis=1,
        result_type="expand"
    )
    results.columns = ["delta", "gamma", "theta", "vega"]
    df = pd.concat([df, results], axis=1)

    # --- OPTION PRICE ---
    df["option_price"] = df.apply(
        lambda row: black_scholes_call(
            S=row["spot"],
            K=row["strike"],
            T=row["T"],
            r=row["r"],
            sigma=row["sigma"]
        ),
        axis=1
    )

    # --- OUTPUT ---
    if output_csv is None:
        output_path = input_path.with_name(f"{input_path.stem}_with_options.csv")
    else:
        output_path = Path(output_csv)

    df.to_csv(output_path, index=False)
    return output_path

# --- SCRIPT ENTRY ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build synthetic options and compute Black-Scholes price & Greeks.")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("-o", "--output", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    output_file = build_synthetic_options(args.input_csv, args.output)
    print(f"Saved synthetic options to: {output_file}")