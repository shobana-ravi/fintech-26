import pandas as pd
import numpy as np
from scipy.stats import norm

HEDGE_BUCKETS = [0.00, 0.25, 0.50, 0.75, 1.00]

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "spy_with_features.csv"
OUTPUT_CSV = "spy_training_dataset.csv"

RISK_FREE_RATE_DEFAULT = 0.03
TX_COST_PER_SHARE = 0.01
HEDGE_PENALTY_LAMBDA = 0.05  # raise to 0.10 or 0.20 if labels still lean too much to extremes


# -----------------------------
# Helpers
# -----------------------------
def black_scholes_call(S, K, T, r, sigma):
    if pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(r) or pd.isna(sigma):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return call_price, delta, gamma, theta, vega


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def choose_best_hedge(row):
    best_score = -np.inf
    best_bucket = np.nan

    for h in HEDGE_BUCKETS:
        suffix = int(h * 100)
        score = row[f"score_{suffix}"]

        if pd.isna(score):
            continue

        if score > best_score:
            best_score = score
            best_bucket = h

    return best_bucket


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT_CSV)
df = normalize_columns(df)

required_cols = ["date", "close", "return_1d", "return_5d", "realized_vol_20d"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print("Available columns:", df.columns.tolist())
    raise ValueError(f"Missing required input columns: {missing}")

# Normalize core columns
df["date"] = pd.to_datetime(df["date"])
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["return_1d"] = pd.to_numeric(df["return_1d"], errors="coerce")
df["return_5d"] = pd.to_numeric(df["return_5d"], errors="coerce")
df["realized_vol_20d"] = pd.to_numeric(df["realized_vol_20d"], errors="coerce")

# -----------------------------
# Synthetic option template
# SPY ATM call, 30 DTE, long 1 contract
# -----------------------------
df["spot"] = df["close"]
df["strike"] = df["spot"].round()
df["dte"] = 30
df["T"] = df["dte"] / 365.0
df["sigma"] = df["realized_vol_20d"]

if "r" not in df.columns:
    df["r"] = RISK_FREE_RATE_DEFAULT
else:
    df["r"] = pd.to_numeric(df["r"], errors="coerce").fillna(RISK_FREE_RATE_DEFAULT)

# -----------------------------
# Current-day Black-Scholes state
# -----------------------------
current_results = df.apply(
    lambda row: black_scholes_call(
        row["spot"],
        row["strike"],
        row["T"],
        row["r"],
        row["sigma"]
    ),
    axis=1
)

df[["option_price", "delta", "gamma", "theta", "vega"]] = pd.DataFrame(
    current_results.tolist(),
    index=df.index
)

# Portfolio greeks for 1 contract
df["portfolio_delta"] = df["delta"] * 100
df["portfolio_gamma"] = df["gamma"] * 100
df["portfolio_theta"] = df["theta"] * 100
df["portfolio_vega"] = df["vega"] * 100

# -----------------------------
# Next-day state for label generation
# -----------------------------
df["spot_next"] = df["spot"].shift(-1)
df["sigma_next"] = df["sigma"].shift(-1)
df["dte_next"] = df["dte"] - 1
df["T_next"] = df["dte_next"] / 365.0

next_results = df.apply(
    lambda row: black_scholes_call(
        row["spot_next"],
        row["strike"],
        row["T_next"],
        row["r"],
        row["sigma_next"]
    ) if pd.notnull(row["spot_next"]) and pd.notnull(row["sigma_next"]) else (np.nan, np.nan, np.nan, np.nan, np.nan),
    axis=1
)

df[["option_price_next", "delta_next", "gamma_next", "theta_next", "vega_next"]] = pd.DataFrame(
    next_results.tolist(),
    index=df.index
)

# Long 1 contract
df["option_pnl_contract"] = (df["option_price_next"] - df["option_price"]) * 100
df["spot_change"] = df["spot_next"] - df["spot"]

# -----------------------------
# Build hedge outcomes for each bucket
# -----------------------------
for h in HEDGE_BUCKETS:
    suffix = int(h * 100)

    hedge_shares_col = f"hedge_shares_{suffix}"
    hedge_pnl_col = f"hedge_pnl_{suffix}"
    hedge_cost_col = f"hedge_cost_{suffix}"
    total_pnl_col = f"total_pnl_{suffix}"
    score_col = f"score_{suffix}"

    df[hedge_shares_col] = -df["portfolio_delta"] * h
    df[hedge_pnl_col] = df[hedge_shares_col] * df["spot_change"]
    df[hedge_cost_col] = abs(df[hedge_shares_col]) * TX_COST_PER_SHARE
    df[total_pnl_col] = (
        df["option_pnl_contract"] +
        df[hedge_pnl_col] -
        df[hedge_cost_col]
    )

    # Better label objective for hedging:
    # minimize the size of the next-day net swing,
    # with a mild penalty for very large hedge sizes
    df[score_col] = (
        -abs(df[total_pnl_col])
        - HEDGE_PENALTY_LAMBDA * abs(df[hedge_shares_col])
    )

# -----------------------------
# Final target label
# -----------------------------
df["target_hedge_ratio_bucket"] = df.apply(choose_best_hedge, axis=1)

ratio_to_class = {
    0.00: 0,
    0.25: 1,
    0.50: 2,
    0.75: 3,
    1.00: 4
}
df["target_class"] = df["target_hedge_ratio_bucket"].map(ratio_to_class)

# -----------------------------
# Final training dataset
# -----------------------------
final_cols = [
    "date",
    "close",
    "return_1d",
    "return_5d",
    "realized_vol_20d",
    "strike",
    "dte",
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
    "target_class",
]

final_df = df[final_cols].copy()

# Drop rows with missing values, especially the last row after shift(-1)
final_df = final_df.dropna().reset_index(drop=True)

# Save final dataset
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved final training dataset to: {OUTPUT_CSV}")
print("\nTarget hedge ratio distribution:")
print(final_df["target_hedge_ratio_bucket"].value_counts().sort_index())
print("\nTarget class distribution:")
print(final_df["target_class"].value_counts().sort_index())
print("\nPreview:")
print(final_df.head())