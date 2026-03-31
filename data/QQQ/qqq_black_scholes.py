import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


CONTRACT_SIZE = 100  # 1 option contract = 100 shares


def black_scholes_call(S, K, T, r, sigma):
    """
    Returns:
    price, delta, gamma, theta, vega
    """

    # Handle edge cases
    if pd.isna(sigma) or sigma == 0 or T == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # --- PRICE ---
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # --- GREEKS ---
    delta = norm.cdf(d1)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    theta = (
        - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )

    vega = S * norm.pdf(d1) * np.sqrt(T)

    return price, delta, gamma, theta, vega


def compute_options(input_csv: str, output_csv: str | None = None) -> Path:
    input_path = Path(input_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    # Remove duplicate columns (keep the first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    print("Loaded file:", input_path.resolve())
    print("Columns:", df.columns.tolist())

    # --- VALIDATION ---
    required_cols = {"spot", "strike", "T", "sigma", "r"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- APPLY BLACK-SCHOLES ---
    results = df.apply(
        lambda row: black_scholes_call(
            S=row["spot"],
            K=row["strike"],
            T=row["T"],
            r=row["r"],
            sigma=row["sigma"],
        ),
        axis=1,
        result_type="expand",
    )

    results.columns = ["option_price", "delta", "gamma", "theta", "vega"]

    df = pd.concat([df, results], axis=1)

    # --- PORTFOLIO GREEKS ---
    df["portfolio_delta"] = df["delta"] * CONTRACT_SIZE
    df["portfolio_gamma"] = df["gamma"] * CONTRACT_SIZE
    df["portfolio_theta"] = df["theta"] * CONTRACT_SIZE
    df["portfolio_vega"] = df["vega"] * CONTRACT_SIZE

    # --- OPTIONAL: convert theta to daily (uncomment if desired) ---
    # df["theta"] = df["theta"] / 365
    # df["portfolio_theta"] = df["theta"] * CONTRACT_SIZE

    # --- CLEAN COLUMN ORDER ---
    cols = list(df.columns)

    greek_cols = [
        "option_price",
        "delta",
        "gamma",
        "theta",
        "vega",
        "portfolio_delta",
        "portfolio_gamma",
        "portfolio_theta",
        "portfolio_vega",
    ]

    ordered_cols = [c for c in cols if c not in greek_cols] + greek_cols
    df = df[ordered_cols]

    # --- OUTPUT ---
    if output_csv is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_with_options.csv"
        )
    else:
        output_path = Path(output_csv)

    df.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Black-Scholes prices, Greeks, and portfolio Greeks."
    )

    parser.add_argument(
        "input_csv",
        nargs="?",
        default="qqq_us_d_with_metrics.csv",
        help="Input CSV with synthetic option inputs",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output CSV path",
    )

    args = parser.parse_args()

    output_file = compute_options(args.input_csv, args.output)

    print(f"Saved output to: {output_file}")