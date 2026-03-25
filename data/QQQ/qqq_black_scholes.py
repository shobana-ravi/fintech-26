import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """
    Returns: price, delta, gamma, theta, vega
    """

    # Avoid division errors
    if sigma == 0 or T == 0:
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
        description="Compute Black-Scholes price and Greeks for synthetic options."
    )

    parser.add_argument(
        "input_csv",
        nargs="?",
        default="qqq_us_d_with_metrics.csv",
        help="Input CSV with option inputs",
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