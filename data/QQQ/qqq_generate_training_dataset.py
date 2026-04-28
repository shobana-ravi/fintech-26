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
HEDGE_PENALTY_LAMBDA = 0.05

# -----------------------------
# Black-Scholes function
# -----------------------------
def black_scholes_call(S, K, T, r, sigma):
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

    # -----------------------------
    # Forward-looking values
    # -----------------------------
    df["spot_next"] = df["close"].shift(-1)
    df["sigma_next"] = df["realized_vol_20d"].shift(-1)

    df["option_price_next"] = np.nan

    for t in range(len(df) - 1):
        df.loc[t, "option_price_next"] = black_scholes_call(
            df.loc[t + 1, "close"],
            df.loc[t, "strike"],
            (df.loc[t, "dte"] - 1) / 365,
            df.loc[t, "r"],
            df.loc[t + 1, "realized_vol_20d"]
        )[0]

    # -----------------------------
    # Option PnL
    # -----------------------------
    df["option_pnl_contract"] = (
        df["option_price_next"] - df["option_price"]
    ) * CONTRACT_SIZE

    # -----------------------------
    # Hedge simulation
    # -----------------------------
    for hedge_ratio in HEDGE_BUCKETS:
        suffix = int(hedge_ratio * 100)

        shares = -df["portfolio_delta"] * hedge_ratio

        df[f"total_pnl_{suffix}"] = (
            df["option_pnl_contract"]
            + shares * (df["spot_next"] - df["close"])
            - np.abs(shares) * COST_PER_SHARE
        )

        df[f"score_{suffix}"] = (
            -np.abs(df[f"total_pnl_{suffix}"])
            - HEDGE_PENALTY_LAMBDA * np.abs(shares)
        )

    # -----------------------------
    # Label selection
    # -----------------------------
    targets = []
    classes = []

    for t in range(len(df)):
        scores = [df.loc[t, f"score_{int(b * 100)}"] for b in HEDGE_BUCKETS]

        if all(pd.isna(scores)):
            targets.append(np.nan)
            classes.append(np.nan)
        else:
            idx = np.nanargmax(scores)
            targets.append(HEDGE_BUCKETS[idx])
            classes.append(idx)

    df["target_hedge_ratio_bucket"] = targets
    df["target_class"] = classes

    # -----------------------------
    # Final dataset (FIXED)
    # -----------------------------
    training_df = df[
        [
            "date",
            "close",
            "spot_next",
            "return_1d",
            "return_5d",
            "realized_vol_20d",
            "sigma_next",
            "strike",
            "dte",
            "option_price",
            "option_price_next",
            "delta",
            "gamma",
            "theta",
            "vega",
            "portfolio_delta",
            "portfolio_gamma",
            "portfolio_theta",
            "portfolio_vega",
            "option_pnl_contract",
            "target_hedge_ratio_bucket",
            "target_class",
        ]
    ].dropna().reset_index(drop=True)

    # -----------------------------
    # Rename to match model
    # -----------------------------
    training_df = training_df.rename(columns={
        "close": "spot_today",
        "dte": "dte_today",
        "option_price": "call_price",
        "option_pnl_contract": "option_pnl",
    })

    # Add T
    training_df["T"] = training_df["dte_today"] / 365

    # -----------------------------
    # Save
    # -----------------------------
    if output_csv is None:
        output_path = input_path.with_name(f"{input_path.stem}_training_dataset.csv")
    else:
        output_path = Path(output_csv)

    training_df.to_csv(output_path, index=False)

    print(f"Training dataset saved to: {output_path}")
    print("\nClass distribution:")
    print(training_df["target_class"].value_counts())

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    generate_training_dataset(args.input_csv, args.output)