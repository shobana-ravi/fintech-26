import pandas as pd
import numpy as np
from scipy.stats import norm
import argparse


def black_scholes_call(S, K, T, r, sigma):
    sigma = np.where(sigma <= 0, 1e-8, sigma)
    T = np.where(T <= 0, 1e-8, T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


def compute_option_pnl(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)

    r = 0.03

    # --- Current values (t) ---
    S_today = df["spot"]
    K = df["strike"]
    price_today = df["call_price"]

    # --- Next day values (t+1) ---
    df["spot_next"] = df["spot"].shift(-1)
    df["sigma_next"] = df["sigma"].shift(-1)

    S_next = df["spot_next"]
    sigma_next = df["sigma_next"]

    T_next = 29 / 365  # fixed as you defined

    # --- Reprice option at t+1 ---
    price_next = black_scholes_call(S_next, K, T_next, r, sigma_next)

    df["call_price_next"] = price_next

    # --- Compute PnL ---
    df["option_pnl"] = price_next - price_today
    df["option_pnl_contract"] = df["option_pnl"] * 100

    # Drop last row (no t+1 data)
    df = df.dropna()

    if output_csv is None:
        output_csv = input_csv.replace(".csv", "_with_pnl.csv")

    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="dia_us_d_with_metrics_with_options_with_hedges.csv",
    )
    parser.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    compute_option_pnl(args.input_csv, args.output)