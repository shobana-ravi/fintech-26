import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
import argparse

# -----------------------------
# Config
# -----------------------------
HEDGE_BUCKETS = [0.00, 0.25, 0.50, 0.75, 1.00]
COST_PER_SHARE = 0.01
CONTRACT_SIZE = 100
HEDGE_PENALTY_LAMBDA = 0.05   # increase to 0.10 or 0.20 if you still get too many 1.00 labels

# -----------------------------
# Black-Scholes function
# -----------------------------
def black_scholes_call(S, K, T, r, sigma):
    """
    Returns: price, delta, gamma, theta, vega
    """
    if pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(r) or pd.isna(sigma):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return price, delta, gamma, theta, vega

# -----------------------------
# Main function
# -----------------------------
def generate_training_dataset(input_csv: str, output_csv=None):
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # -----------------------------
    # Standardize column names
    # -----------------------------
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)

    required_cols = [
        "date",
        "close",
        "return_1d",
        "return_5d",
        "realized_vol_20d",
        "strike",
        "dte",
        "r",
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
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -----------------------------
    # Step 7: compute next-day option price
    # -----------------------------
    df["option_price_next"] = np.nan
    df["spot_next"] = df["close"].shift(-1)
    df["sigma_next"] = df["realized_vol_20d"].shift(-1)

    for t in range(len(df) - 1):
        spot_next = df.loc[t + 1, "close"]
        K = df.loc[t, "strike"]
        T_next = (df.loc[t, "dte"] - 1) / 365
        r = df.loc[t, "r"]
        sigma_next = df.loc[t + 1, "realized_vol_20d"]

        df.loc[t, "option_price_next"] = black_scholes_call(
            spot_next, K, T_next, r, sigma_next
        )[0]

    # -----------------------------
    # Step 8: option P&L
    # -----------------------------
    df["option_pnl_contract"] = (df["option_price_next"] - df["option_price"]) * CONTRACT_SIZE
    df["option_pnl"] = df["option_pnl_contract"] / CONTRACT_SIZE

    # -----------------------------
    # Step 8: compute hedge outcomes
    # -----------------------------
    for hedge_ratio in HEDGE_BUCKETS:
        suffix = int(hedge_ratio * 100)

        # IMPORTANT:
        # portfolio_delta is already contract-scaled in your pipeline
        hedge_shares_col = f"hedge_shares_{suffix}"
        hedge_pnl_col = f"hedge_pnl_{suffix}"
        hedge_cost_col = f"hedge_cost_{suffix}"
        total_pnl_col = f"total_pnl_{suffix}"
        score_col = f"score_{suffix}"

        df[hedge_shares_col] = -df["portfolio_delta"] * hedge_ratio
        df[hedge_pnl_col] = df[hedge_shares_col] * (df["spot_next"] - df["close"])
        df[hedge_cost_col] = np.abs(df[hedge_shares_col]) * COST_PER_SHARE
        df[total_pnl_col] = df["option_pnl_contract"] + df[hedge_pnl_col] - df[hedge_cost_col]

        # NEW:
        # choose hedge that reduces next-day net swing,
        # with a mild penalty for oversized hedges
        df[score_col] = (
            -np.abs(df[total_pnl_col])
            - HEDGE_PENALTY_LAMBDA * np.abs(df[hedge_shares_col])
        )

    # -----------------------------
    # Step 9: select best hedge ratio
    # -----------------------------
    target_buckets = []
    target_classes = []

    for t in range(len(df)):
        scores_today = [df.loc[t, f"score_{int(bucket * 100)}"] for bucket in HEDGE_BUCKETS]

        if all(pd.isna(scores_today)):
            target_buckets.append(np.nan)
            target_classes.append(np.nan)
        else:
            best_idx = np.nanargmax(scores_today)
            target_buckets.append(HEDGE_BUCKETS[best_idx])
            target_classes.append(best_idx)

    df["target_hedge_ratio_bucket"] = target_buckets
    df["target_class"] = target_classes

    # -----------------------------
    # Step 10: build final training dataset (schema matches models/XGBoost.py)
    # -----------------------------
    df["spot_today"] = df["close"]
    df["dte_today"] = df["dte"]
    df["call_price"] = df["option_price"]
    df["T"] = df["dte_today"] / 365.0

    final_columns = [
        "date",
        "spot_today",
        "return_1d",
        "return_5d",
        "realized_vol_20d",
        "sigma_next",
        "strike",
        "T",
        "dte_today",
        "call_price",
        "delta",
        "gamma",
        "theta",
        "vega",
        "portfolio_delta",
        "portfolio_gamma",
        "portfolio_theta",
        "portfolio_vega",
        "option_pnl",
        "target_hedge_ratio_bucket",
        "target_class",
    ]

    training_df = df[final_columns].dropna().reset_index(drop=True)

    if output_csv is None:
        output_path = input_path.with_name(f"{input_path.stem}_training_dataset.csv")
    else:
        output_path = Path(output_csv)

    training_df.to_csv(output_path, index=False)
    print(f"Training dataset saved to: {output_path}")

    print("\nTarget hedge ratio distribution:")
    print(training_df["target_hedge_ratio_bucket"].value_counts().sort_index())

    print("\nTarget class distribution:")
    print(training_df["target_class"].value_counts().sort_index())

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training dataset with hedge labels")
    parser.add_argument("input_csv", help="CSV file with portfolio Greeks")
    parser.add_argument("-o", "--output", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    generate_training_dataset(args.input_csv, args.output)