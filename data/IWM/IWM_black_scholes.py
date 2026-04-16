import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
import argparse


def black_scholes_call(S, K, T, r, sigma):
    """
    Returns: price, delta, gamma, theta, vega for a European call
    """
    sigma = np.where(sigma <= 0, 1e-8, sigma)
    T = np.where(T <= 0, 1e-8, T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )

    return price, delta, gamma, theta, vega


def compute_option_metrics(input_csv: str, output_csv: str | None = None):
    df = pd.read_csv(input_csv)

    # Create option inputs from your dataset
    df["spot"] = df["close"]

    # ATM strike assumption (rounded)
    df["strike"] = df["close"].round()

    # days to expiration
    df["dte"] = 30

    # time to maturity
    df["T"] = df["dte"] / 365

    # volatility estimate
    df["sigma"] = df["realized_vol_20d"]

    # risk free rate
    df["r"] = 0.03

    # option type
    df["option_type"] = "call"

    S = df["spot"].values
    K = df["strike"].values
    T = df["T"].values
    sigma = df["sigma"].values
    r = df["r"].values

    price, delta, gamma, theta, vega = black_scholes_call(S, K, T, r, sigma)

    df["call_price"] = price
    df["delta"] = delta
    df["gamma"] = gamma
    df["theta"] = theta
    df["vega"] = vega

    if output_csv is None:
        output_csv = input_csv.replace(".csv", "_with_options.csv")

    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Black-Scholes option price and Greeks for IWM dataset"
    )

    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "iwm_us_d_with_metrics.csv"),
        help="Path to IWM metrics CSV (default: iwm_us_d_with_metrics.csv)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output CSV path",
    )

    args = parser.parse_args()

    compute_option_metrics(args.input_csv, args.output)