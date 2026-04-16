import pandas as pd
import numpy as np
from scipy.stats import norm
import argparse


# --- Constants ---
HEDGE_BUCKETS = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
COST_PER_SHARE = 0.01
RISK_FREE_RATE = 0.03
CONTRACT_SIZE = 100
HEDGE_PENALTY_LAMBDA = 0.05   # raise to 0.10 or 0.20 if needed


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
    df.columns = [c.strip() for c in df.columns]

    required = ["spot", "sigma", "strike", "call_price", "delta"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Step 7: Reprice next day ---
    df["spot_next"] = df["spot"].shift(-1)
    df["sigma_next"] = df["sigma"].shift(-1)

    T_next = 29 / 365

    # Recompute option price at t+1
    df["call_price_next"] = black_scholes_call(
        df["spot_next"],
        df["strike"],
        T_next,
        RISK_FREE_RATE,
        df["sigma_next"]
    )

    # Option PnL for 1 contract
    df["option_pnl"] = df["call_price_next"] - df["call_price"]
    df["option_pnl_contract"] = df["option_pnl"] * CONTRACT_SIZE

    # Portfolio greeks for 1 contract
    df["portfolio_delta"] = df["delta"] * CONTRACT_SIZE

    if "gamma" in df.columns:
        df["portfolio_gamma"] = df["gamma"] * CONTRACT_SIZE
    if "theta" in df.columns:
        df["portfolio_theta"] = df["theta"] * CONTRACT_SIZE
    if "vega" in df.columns:
        df["portfolio_vega"] = df["vega"] * CONTRACT_SIZE

    # --- Step 8 & 9: Hedge PnL + Label ---
    total_pnl_cols = []
    score_cols = []

    for ratio in HEDGE_BUCKETS:
        suffix = int(ratio * 100)

        hedge_shares_col = f"hedge_shares_{suffix}"
        hedge_pnl_col = f"hedge_pnl_{suffix}"
        hedge_cost_col = f"hedge_cost_{suffix}"
        total_pnl_col = f"total_pnl_{suffix}"
        score_col = f"score_{suffix}"

        # IMPORTANT: use contract-scaled delta exposure
        df[hedge_shares_col] = -df["portfolio_delta"] * ratio

        df[hedge_pnl_col] = df[hedge_shares_col] * (df["spot_next"] - df["spot"])
        df[hedge_cost_col] = np.abs(df[hedge_shares_col]) * COST_PER_SHARE

        df[total_pnl_col] = (
            df["option_pnl_contract"] +
            df[hedge_pnl_col] -
            df[hedge_cost_col]
        )

        # Better target score:
        # reduce next-day P&L swing and mildly penalize oversized hedges
        df[score_col] = (
            -np.abs(df[total_pnl_col])
            - HEDGE_PENALTY_LAMBDA * np.abs(df[hedge_shares_col])
        )

        total_pnl_cols.append(total_pnl_col)
        score_cols.append(score_col)

    # Best hedge: choose highest score, NOT highest raw pnl
    df["best_hedge_idx"] = df[score_cols].values.argmax(axis=1)
    df["target_hedge_ratio_bucket"] = HEDGE_BUCKETS[df["best_hedge_idx"]]
    df["target_class"] = df["best_hedge_idx"]

    # Drop last row (no next-day data)
    df = df.dropna().reset_index(drop=True)

    # --- Step 10: Build final ML dataset ---
    df = df.rename(columns={
        "spot": "close",
        "call_price": "option_price"
    })

    final_columns = [
        "date",
        "close",
        "return_1d",
        "return_5d",
        "realized_vol_20d",
        "strike",
        "T",
        "option_price",
        "delta",
        "gamma",
        "theta",
        "vega",
        "portfolio_delta",
        "portfolio_gamma",
        "portfolio_theta",
        "portfolio_vega",
        "target_hedge_ratio_bucket",
        "target_class"
    ]

    final_columns = [col for col in final_columns if col in df.columns]
    df_final = df[final_columns].copy()

    # Save output
    if output_csv is None:
        output_csv = "dia_ml_dataset_fixed.csv"

    df_final.to_csv(output_csv, index=False)
    print(f"Final ML dataset saved: {output_csv}")

    print("\nTarget hedge ratio distribution:")
    print(df_final["target_hedge_ratio_bucket"].value_counts().sort_index())

    print("\nTarget class distribution:")
    print(df_final["target_class"].value_counts().sort_index())


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full pipeline: option PnL, hedge simulation, and ML dataset for DIA"
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