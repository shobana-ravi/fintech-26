import pandas as pd
import numpy as np
from scipy.stats import norm
import argparse


# --- Constants ---
HEDGE_BUCKETS = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
COST_PER_SHARE = 0.01


# --- Black-Scholes Call ---
def black_scholes_call(S, K, T, r, sigma):
    sigma = np.where(sigma <= 0, 1e-8, sigma)
    T = np.where(T <= 0, 1e-8, T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


# --- Main Pipeline ---
def compute_option_pnl(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)

    r = 0.03

    # --- Step 7: Reprice next day ---
    df["spot_next"] = df["spot"].shift(-1)
    df["sigma_next"] = df["sigma"].shift(-1)

    S_today = df["spot"]
    S_next = df["spot_next"]
    K = df["strike"]
    sigma_next = df["sigma_next"]

    T_next = 29 / 365

    # Recompute option price at t+1
    price_next = black_scholes_call(S_next, K, T_next, r, sigma_next)

    df["call_price_next"] = price_next

    # Option PnL
    df["option_pnl"] = df["call_price_next"] - df["call_price"]
    df["option_pnl_contract"] = df["option_pnl"] * 100

    # --- Step 8 & 9: Hedge PnL + Label ---
    total_pnl_cols = []

    for ratio in HEDGE_BUCKETS:
        suffix = int(ratio * 100)

        # Hedge shares
        hedge_shares = -df["delta"] * ratio

        # Hedge PnL
        hedge_pnl = hedge_shares * (df["spot_next"] - df["spot"])

        # Transaction cost
        hedge_cost = np.abs(hedge_shares) * COST_PER_SHARE

        # Total PnL
        total_pnl = df["option_pnl_contract"] + hedge_pnl - hedge_cost

        col_name = f"total_pnl_{suffix}"
        df[col_name] = total_pnl
        total_pnl_cols.append(col_name)

    # --- Choose best hedge ---
    df["best_hedge_idx"] = df[total_pnl_cols].values.argmax(axis=1)
    df["target_hedge_ratio"] = HEDGE_BUCKETS[df["best_hedge_idx"]]

    # Drop last row (no t+1 data)
    df = df.dropna()

    # Output
    if output_csv is None:
        output_csv = input_csv.replace(".csv", "_final.csv")

    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute option PnL, hedge PnL, and optimal hedge label for DIA"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="dia_us_d_with_metrics_with_options_with_hedges.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output CSV path",
    )

    args = parser.parse_args()

    compute_option_pnl(args.input_csv, args.output)