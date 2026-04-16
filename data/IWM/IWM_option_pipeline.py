import pandas as pd
import numpy as np
from scipy.stats import norm
import argparse


# --------------------------------------------------
# Constants
# --------------------------------------------------

HEDGE_BUCKETS = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
COST_PER_SHARE = 0.01
RISK_FREE_RATE = 0.03
HEDGE_PENALTY_LAMBDA = 0.05


# --------------------------------------------------
# Black-Scholes Call Price
# --------------------------------------------------

def black_scholes_call(S, K, T, r, sigma):
    sigma = np.where(sigma <= 0, 1e-8, sigma)
    T = np.where(T <= 0, 1e-8, T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def compute_option_pipeline(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]

    print("Detected columns:", df.columns.tolist())

    # Accept either close or spot_today
    if "close" not in df.columns and "spot_today" not in df.columns:
        raise ValueError("Need either 'close' or 'spot_today' column.")

    if "close" not in df.columns and "spot_today" in df.columns:
        df["close"] = df["spot_today"]

    required = [
        "date",
        "close",
        "return_1d",
        "return_5d",
        "realized_vol_20d",
        "strike",
        "T",
        "call_price",
        "delta",
        "gamma",
        "theta",
        "vega",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---------------------------------------------
    # Base state
    # ---------------------------------------------

    df["cost_per_share"] = COST_PER_SHARE

    df["spot_today"] = df["close"]

    if "spot_next" not in df.columns:
        df["spot_next"] = df["close"].shift(-1)

    if "sigma_next" not in df.columns:
        df["sigma_next"] = df["realized_vol_20d"].shift(-1)

    if "dte_today" not in df.columns:
        df["dte_today"] = np.round(df["T"] * 365).astype(int)

    if "dte_next" not in df.columns:
        df["dte_next"] = df["dte_today"] - 1

    if "T_next" not in df.columns:
        df["T_next"] = df["dte_next"] / 365.0

    # Portfolio greeks for 1 contract
    df["portfolio_delta"] = df["delta"] * 100
    df["portfolio_gamma"] = df["gamma"] * 100
    df["portfolio_theta"] = df["theta"] * 100
    df["portfolio_vega"] = df["vega"] * 100

    # ---------------------------------------------
    # Reprice option next day if missing
    # ---------------------------------------------

    if "call_price_next" not in df.columns:
        df["call_price_next"] = black_scholes_call(
            df["spot_next"],
            df["strike"],
            df["T_next"],
            RISK_FREE_RATE,
            df["sigma_next"]
        )

    # ---------------------------------------------
    # Option PnL if missing
    # ---------------------------------------------

    if "option_pnl" not in df.columns:
        df["option_pnl"] = df["call_price_next"] - df["call_price"]

    if "option_pnl_contract" not in df.columns:
        df["option_pnl_contract"] = df["option_pnl"] * 100

    df["spot_change"] = df["spot_next"] - df["spot_today"]

    # ---------------------------------------------
    # Hedge calculations
    # ---------------------------------------------

    score_cols = []

    for ratio in HEDGE_BUCKETS:
        suffix = int(ratio * 100)

        hedge_shares_col = f"hedge_shares_{suffix}"
        hedge_pnl_col = f"hedge_pnl_{suffix}"
        hedge_cost_col = f"hedge_cost_{suffix}"
        total_pnl_col = f"total_pnl_{suffix}"
        score_col = f"score_{suffix}"

        # Use contract delta exposure
        df[hedge_shares_col] = -df["portfolio_delta"] * ratio

        df[hedge_pnl_col] = df[hedge_shares_col] * df["spot_change"]
        df[hedge_cost_col] = np.abs(df[hedge_shares_col]) * df["cost_per_share"]

        df[total_pnl_col] = (
            df["option_pnl_contract"]
            + df[hedge_pnl_col]
            - df[hedge_cost_col]
        )

        # Risk-reduction score instead of raw profit max
        df[score_col] = (
            -np.abs(df[total_pnl_col])
            - HEDGE_PENALTY_LAMBDA * np.abs(df[hedge_shares_col])
        )

        score_cols.append(score_col)

    # ---------------------------------------------
    # Label creation
    # ---------------------------------------------

    df["best_hedge_idx"] = df[score_cols].values.argmax(axis=1)
    df["target_hedge_ratio_bucket"] = HEDGE_BUCKETS[df["best_hedge_idx"]]
    df["target_class"] = df["best_hedge_idx"]

    # Remove last row / incomplete rows
    df = df.dropna().reset_index(drop=True)

    # ---------------------------------------------
    # Final dataset columns
    # ---------------------------------------------

    final_columns = [
        "date",
        "spot_today",
        "spot_next",
        "return_1d",
        "return_5d",
        "realized_vol_20d",
        "sigma_next",
        "strike",
        "T",
        "dte_today",
        "dte_next",
        "T_next",
        "call_price",
        "call_price_next",
        "delta",
        "gamma",
        "theta",
        "vega",
        "portfolio_delta",
        "portfolio_gamma",
        "portfolio_theta",
        "portfolio_vega",
        "option_pnl",
        "option_pnl_contract",
        "cost_per_share",
        "hedge_shares_0",
        "hedge_shares_25",
        "hedge_shares_50",
        "hedge_shares_75",
        "hedge_shares_100",
        "total_pnl_0",
        "total_pnl_25",
        "total_pnl_50",
        "total_pnl_75",
        "total_pnl_100",
        "score_0",
        "score_25",
        "score_50",
        "score_75",
        "score_100",
        "best_hedge_idx",
        "target_hedge_ratio_bucket",
        "target_class",
    ]

    final_columns = [c for c in final_columns if c in df.columns]
    df_final = df[final_columns].copy()

    if output_csv is None:
        output_csv = "iwm_ml_dataset.csv"

    df_final.to_csv(output_csv, index=False)

    print("Final ML dataset saved:", output_csv)
    print("\nTarget hedge ratio distribution:")
    print(df_final["target_hedge_ratio_bucket"].value_counts().sort_index())
    print("\nTarget class distribution:")
    print(df_final["target_class"].value_counts().sort_index())


# --------------------------------------------------
# CLI
# --------------------------------------------------

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