import pandas as pd
import numpy as np
from scipy.stats import norm
HEDGE_BUCKETS = [0.00, 0.25, 0.50, 0.75, 1.00]

# Load dataset
df = pd.read_csv("spy_with_features.csv")

# Black-Scholes formulas
def black_scholes_call(S, K, T, r, sigma):
    if sigma == 0 or T == 0:
        return 0, 0, 0, 0, 0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return call_price, delta, gamma, theta, vega

# Apply function row-wise
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

df[["option_price", "delta", "gamma", "theta", "vega"]] = pd.DataFrame(results.tolist(), index=df.index)
df["portfolio_delta"] = df["delta"] * 100
df["portfolio_gamma"] = df["gamma"] * 100
df["portfolio_theta"] = df["theta"] * 100
df["portfolio_vega"] = df["vega"] * 100
for h in HEDGE_BUCKETS:
    col_name = f"hedge_shares_{int(h*100)}"
    df[col_name] = -df["portfolio_delta"] * h
    df["spot_next"] = df["spot"].shift(-1)
df["sigma_next"] = df["sigma"].shift(-1)

# DTE goes from 30 → 29
df["dte_next"] = 29
df["T_next"] = 29 / 365
df["option_price_next"] = df.apply(
    lambda row: black_scholes_call(
        row["spot_next"],
        row["strike"],
        row["T_next"],
        row["r"],
        row["sigma_next"]
    )[0] if pd.notnull(row["spot_next"]) else None,
    axis=1
)
df["option_pnl"] = df["option_price_next"] - df["option_price"]
df["option_pnl_contract"] = df["option_pnl"] * 100
# Transaction cost per share
cost_per_share = 0.01

# Price change
df["spot_change"] = df["spot_next"] - df["spot"]

# Compute hedge P&L for each hedge bucket

for h in HEDGE_BUCKETS:
    suffix = int(h * 100)
    
    hedge_col = f"hedge_shares_{suffix}"
    
    # Hedge P&L
    df[f"hedge_pnl_{suffix}"] = df[hedge_col] * df["spot_change"]
    
    # Transaction cost
    df[f"hedge_cost_{suffix}"] = abs(df[hedge_col]) * cost_per_share
    
    # Net hedge P&L after cost
    df[f"net_hedge_pnl_{suffix}"] = (
        df[f"hedge_pnl_{suffix}"] - df[f"hedge_cost_{suffix}"]
    )
df.to_csv("spy_with_greeks.csv", index=False)
# Save output
df.to_csv("spy_black_scholes.csv", index=False)

print("Done! File saved as data/SPY/spy_with_greeks.csv")
