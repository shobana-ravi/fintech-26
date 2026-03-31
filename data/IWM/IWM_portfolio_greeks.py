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
        - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )

    return price, delta, gamma, theta, vega


def compute_option_metrics(input_csv: str, output_csv: str | None = None):

    df = pd.read_csv(input_csv)

    required_cols = ["spot", "strike", "T", "sigma"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    S = df["spot"].values
    K = df["strike"].values
    T = df["T"].values
    sigma = df["sigma"].values
    r = 0.03

    price, delta, gamma, theta, vega = black_scholes_call(S, K, T, r, sigma)

    # Option Greeks
    df["call_price"] = price
    df["delta"] = delta
    df["gamma"] = gamma
    df["theta"] = theta
    df["vega"] = vega

    # Portfolio Greeks (1 option contract = 100 shares)
    df["portfolio_delta"] = df["delta"] * 100
    df["portfolio_gamma"] = df["gamma"] * 100
    df["portfolio_theta"] = df["theta"] * 100
    df["portfolio_vega"] = df["vega"] * 100

    if output_csv is None:
        output_csv = input_csv.replace(".csv", "_with_options.csv")

    df.to_csv(output_csv, index=False)

    print("Saved:", output_csv)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute Black-Scholes option price and portfolio Greeks for IWM dataset"
    )

    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "iwm_us_d_with_metrics.csv"),
        help="Path to IWM metrics CSV",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output CSV path",
    )

    args = parser.parse_args()

    compute_option_metrics(args.input_csv, args.output)