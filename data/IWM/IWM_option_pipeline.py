import pandas as pd
import numpy as np
from scipy.stats import norm
import argparse


# Constants
HEDGE_BUCKETS = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
COST_PER_SHARE = 0.01
RISK_FREE_RATE = 0.03


# Black-Scholes Call
def black_scholes_call(S, K, T, r, sigma):

    sigma = np.where(sigma <= 0, 1e-8, sigma)
    T = np.where(T <= 0, 1e-8, T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Main pipeline
def compute_option_pipeline(input_csv, output_csv=None):

    df = pd.read_csv(input_csv)

    print("Detected columns:", df.columns.tolist())


    # ------------------------------------------------
    # STEP 7 — Reprice option on next day
    # ------------------------------------------------

    df["spot_next"] = df["close"].shift(-1)

    df["sigma_next"] = df["realized_vol_20d"].shift(-1)

    df["dte_next"] = 29

    df["T_next"] = df["dte_next"] / 365


    df["call_price_next"] = black_scholes_call(
        df["spot_next"],
        df["strike"],
        df["T_next"],
        RISK_FREE_RATE,
        df["sigma_next"]
    )


    # Option PnL
    df["option_pnl"] = df["call_price_next"] - df["call_price"]

    df["option_pnl_contract"] = df["option_pnl"] * 100


    # ------------------------------------------------
    # Hedge simulation
    # ------------------------------------------------

    total_pnl_cols = []

    for ratio in HEDGE_BUCKETS:

        suffix = int(ratio * 100)

        hedge_shares = -df["delta"] * ratio

        hedge_pnl = hedge_shares * (df["spot_next"] - df["close"])

        hedge_cost = np.abs(hedge_shares) * COST_PER_SHARE

        total_pnl = df["option_pnl_contract"] + hedge_pnl - hedge_cost

        col = f"total_pnl_{suffix}"

        df[col] = total_pnl

        total_pnl_cols.append(col)


    # Best hedge ratio
    df["best_hedge_idx"] = df[total_pnl_cols].values.argmax(axis=1)

    df["target_hedge_ratio_bucket"] = HEDGE_BUCKETS[df["best_hedge_idx"]]


    # Remove last row (no t+1 data)
    df = df.dropna()


    # ------------------------------------------------
    # Final ML Dataset
    # ------------------------------------------------

    final_columns = [

        # Date + underlying
        "date",
        "close",
        "spot_next",

        # Returns + volatility
        "return_1d",
        "return_5d",
        "realized_vol_20d",
        "sigma_next",

        # Option parameters
        "strike",
        "T",
        "dte_next",
        "T_next",

        # Option pricing
        "call_price",
        "call_price_next",

        # Greeks
        "delta",
        "gamma",
        "theta",
        "vega",

        # PnL
        "option_pnl",
        "option_pnl_contract",

        # ML label
        "target_hedge_ratio_bucket"
    ]

    final_columns = [c for c in final_columns if c in df.columns]

    df_final = df[final_columns].copy()


    if output_csv is None:
        output_csv = "iwm_ml_dataset.csv"


    df_final.to_csv(output_csv, index=False)

    print("Final ML dataset saved:", output_csv)


# CLI
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="IWM Option Repricing + Hedge Simulation + ML Dataset"
    )

    parser.add_argument(
        "input_csv",
        help="Input CSV with option greeks"
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None
    )

    args = parser.parse_args()

    compute_option_pipeline(args.input_csv, args.output)