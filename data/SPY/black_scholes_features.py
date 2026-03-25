import pandas as pd
import numpy as np
from scipy.stats import norm

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
# Save output
df.to_csv("spy_black_scholes.csv", index=False)

print("Done! File saved as data/SPY/spy_with_greeks.csv")
