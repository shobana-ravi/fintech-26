import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/spy_us_d.csv")
df.columns = df.columns.str.lower()

# Sort by date if exists
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# Returns
df["return_1d"] = df["close"] / df["close"].shift(1) - 1
df["return_5d"] = df["close"] / df["close"].shift(5) - 1

# Daily returns for volatility
df["daily_return"] = df["close"].pct_change()

# Realized volatility (20-day rolling)
df["realized_vol_20d"] = df["daily_return"].rolling(20).std() * np.sqrt(252)

# -----------------------------
# Synthetic Option Variables
# -----------------------------

# spot price
df["spot"] = df["close"]

# nearest ATM strike
df["strike"] = df["spot"].round()

# days to expiration
df["dte"] = 30

# time to maturity
df["T"] = 30 / 365

# volatility
df["sigma"] = df["realized_vol_20d"]

# risk-free rate
df["r"] = 0.03
 
# option type
df["option_type"] = "call"

# Save output
df.to_csv("data/spy_with_features.csv", index=False)

print("Done! File saved as data/spy_with_features.csv")
