import pandas as pd
import numpy as np
from scipy.stats import norm

HEDGE_BUCKETS = [0.00, 0.25, 0.50, 0.75, 1.00]

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("spy_with_features.csv")

# -----------------------------
# Black-Scholes formulas
# -----------------------------
def black_scholes_call(S, K, T, r, sigma):
    if pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(r) or pd.isna(sigma):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
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

# -----------------------------
# Current-day option state
# -----------------------------
results = df.apply(
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
    results.tolist(), index=df.index
)

# Portfolio greeks for 1 contract
df["portfolio_delta"] = df["delta"] * 100
df["portfolio_gamma"] = df["gamma"] * 100
df["portfolio_theta"] = df["theta"] * 100
df["portfolio_vega"] = df["vega"] * 100

# -----------------------------
# Create hedge-share columns
# -----------------------------
for h in HEDGE_BUCKETS:
    suffix = int(h * 100)
    df[f"hedge_shares_{suffix}"] = -df["portfolio_delta"] * h

# -----------------------------
# Next-day market state
# -----------------------------
df["spot_next"] = df["spot"].shift(-1)
df["sigma_next"] = df["sigma"].shift(-1)

# DTE goes from 30 -> 29
df["dte_next"] = 29
df["T_next"] = 29 / 365

# Reprice next-day option using same strike, next-day spot, next-day sigma
df["option_price_next"] = df.apply(
    lambda row: black_scholes_call(
        row["spot_next"],
        row["strike"],
        row["T_next"],
        row["r"],
        row["sigma_next"]
    )[0]
    if pd.notnull(row["spot_next"]) and pd.notnull(row["sigma_next"])
    else np.nan,
    axis=1
)

# Option P&L for 1 contract
df["option_pnl"] = df["option_price_next"] - df["option_price"]
df["option_pnl_contract"] = df["option_pnl"] * 100

# Underlying change
df["spot_change"] = df["spot_next"] - df["spot"]

# -----------------------------
# Hedge P&L and costs
# -----------------------------
cost_per_share = 0.01
hedge_penalty_lambda = 0.03   # tune this; larger = more preference for partial hedges

for h in HEDGE_BUCKETS:
    suffix = int(h * 100)
    hedge_col = f"hedge_shares_{suffix}"

    df[f"hedge_pnl_{suffix}"] = df[hedge_col] * df["spot_change"]
    df[f"hedge_cost_{suffix}"] = abs(df[hedge_col]) * cost_per_share

    # Net total P&L after hedge
    df[f"total_pnl_{suffix}"] = (
        df["option_pnl_contract"]
        + df[f"hedge_pnl_{suffix}"]
        - df[f"hedge_cost_{suffix}"]
    )

    # Risk-adjusted score:
    # prefer smaller absolute net P&L swing, and penalize larger hedge size
    df[f"score_{suffix}"] = (
        -abs(df[f"total_pnl_{suffix}"])
        - hedge_penalty_lambda * abs(df[hedge_col])
    )

# -----------------------------
# Choose best hedge bucket
# -----------------------------
def choose_best_hedge(row):
    best_score = -np.inf
    best_hedge = np.nan
    best_idx = np.nan

    for idx, h in enumerate(HEDGE_BUCKETS):
        suffix = int(h * 100)
        score = row[f"score_{suffix}"]

        if pd.isna(score):
            continue

        if score > best_score:
            best_score = score
            best_hedge = h
            best_idx = idx

    return pd.Series([best_idx, best_hedge])

df[["best_hedge_idx", "target_hedge_ratio_bucket"]] = df.apply(
    choose_best_hedge, axis=1
)

# Optional integer classes for XGBoost classifier
ratio_to_class = {
    0.00: 0,
    0.25: 1,
    0.50: 2,
    0.75: 3,
    1.00: 4
}

df["target_class"] = df["target_hedge_ratio_bucket"].map(ratio_to_class)

# -----------------------------
# Save outputs
# -----------------------------
df.to_csv("spy_with_greeks.csv", index=False)
df.to_csv("spy_black_scholes.csv", index=False)

print("Done! Saved:")
print(" - spy_with_greeks.csv")
print(" - spy_black_scholes.csv")

# Quick check of class distribution
print("\nTarget hedge ratio distribution:")
print(df["target_hedge_ratio_bucket"].value_counts(dropna=False).sort_index())