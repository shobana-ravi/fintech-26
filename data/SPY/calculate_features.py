import pandas as pd
import numpy as np

# Load CSV (make sure the file is in the same directory)
df = pd.read_csv("spy_us_d.csv")

# Ensure correct column name
# Adjust if your column is named differently (e.g., 'Close' instead of 'close')
df.columns = df.columns.str.lower()

# Sort by date if date column exists
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# Calculate returns
df["return_1d"] = df["close"] / df["close"].shift(1) - 1
df["return_5d"] = df["close"] / df["close"].shift(5) - 1

# Daily returns for volatility
df["daily_return"] = df["close"].pct_change()

# Realized volatility (20-day rolling)
df["realized_vol_20d"] = df["daily_return"].rolling(20).std() * np.sqrt(252)

# Synthetic option features

# spot price
df["spot"] = df["close"]

# nearest ATM strike (round to nearest integer)
df["strike"] = df["spot"].round()

# days to expiration
df["dte"] = 30

# time to maturity (in years)
df["T"] = 30 / 365

# volatility (already computed)
df["sigma"] = df["realized_vol_20d"]

# risk-free rate
df["r"] = 0.03

# option type
df["option_type"] = "call"

# Save output
df.to_csv("spy_with_features.csv", index=False)

print("Done! File saved as spy_with_features.csv")
