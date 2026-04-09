import pandas as pd
import numpy as np

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
DATA_PATH = "qqq_us_d_with_synthetic_data_with_greeks_portfolio_training_dataset.csv"  # adjust if needed
HEDGE_BUCKETS = [0.00, 0.25, 0.50, 0.75, 1.00]

# --------------------------------------------------------------------
# Load & normalize columns
# --------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Normalize column names to avoid Close vs close mismatches
df.columns = df.columns.str.strip().str.lower()

print("Columns in dataset:", df.columns.tolist())
print("Dataset shape:", df.shape)

# Expected schema
expected_cols = {
    "date", "close",
    "return_1d", "return_5d", "realized_vol_20d",
    "strike", "dte",
    "option_price", "delta", "gamma", "theta", "vega",
    "portfolio_delta", "portfolio_gamma", "portfolio_theta", "portfolio_vega",
    "target_hedge_ratio_bucket",
}

missing = expected_cols - set(df.columns)
extra = set(df.columns) - expected_cols

if missing:
    raise ValueError(f"Missing expected columns: {sorted(missing)}")
if extra:
    print("Warning: Extra columns present:", sorted(extra))

# --------------------------------------------------------------------
# Basic data health
# --------------------------------------------------------------------
print("\nMissing values per column:")
print(df.isna().sum())

print("\nFirst 5 rows:")
print(df.head())

# Check date is monotonic increasing (daily time series)
if not pd.to_datetime(df["date"]).is_monotonic_increasing:
    print("\nWarning: 'date' column is not strictly monotonically increasing.")

# --------------------------------------------------------------------
# Option Greeks consistency checks
# --------------------------------------------------------------------
print("\n--- Option Greeks Checks ---")

# Prices should be positive
n_price_le_zero = (df["option_price"] <= 0).sum()
print("Option price <= 0:", n_price_le_zero)

# Vanilla call delta in [0,1] for long position
n_delta_outside = ((df["delta"] < 0) | (df["delta"] > 1)).sum()
print("Delta outside [0,1]:", n_delta_outside)

# Gamma should be >= 0 for vanilla options
n_gamma_negative = (df["gamma"] < 0).sum()
print("Gamma negative:", n_gamma_negative)

# Vega should be >= 0 for vanilla options
n_vega_negative = (df["vega"] < 0).sum()
print("Vega negative:", n_vega_negative)

# --------------------------------------------------------------------
# Portfolio Greeks scaling checks (contract size = 100)
# --------------------------------------------------------------------
print("\n--- Portfolio Greeks Checks ---")

delta_diff = (df["portfolio_delta"] - df["delta"] * 100).abs().max()
gamma_diff = (df["portfolio_gamma"] - df["gamma"] * 100).abs().max()
theta_diff = (df["portfolio_theta"] - df["theta"] * 100).abs().max()
vega_diff = (df["portfolio_vega"] - df["vega"] * 100).abs().max()

print("Max |portfolio_delta - delta * 100|:", delta_diff)
print("Max |portfolio_gamma - gamma * 100|:", gamma_diff)
print("Max |portfolio_theta - theta * 100|:", theta_diff)
print("Max |portfolio_vega - vega * 100|:", vega_diff)

# --------------------------------------------------------------------
# Target hedge ratio bucket checks
# --------------------------------------------------------------------
print("\n--- Hedge Ratio Label Checks ---")

print("Raw bucket distribution (including NaN):")
print(df["target_hedge_ratio_bucket"].value_counts(dropna=False).sort_index())

# Check allowed values
valid_bucket_set = set(HEDGE_BUCKETS)

invalid_mask = (
    df["target_hedge_ratio_bucket"].notna()
    & ~df["target_hedge_ratio_bucket"].isin(valid_bucket_set)
)
n_invalid = invalid_mask.sum()
print("\nInvalid hedge labels (not in HEDGE_BUCKETS):", n_invalid)

if n_invalid > 0:
    print("Example invalid labels:")
    print(df.loc[invalid_mask, "target_hedge_ratio_bucket"].head(10))

# Missing labels
n_missing_labels = df["target_hedge_ratio_bucket"].isna().sum()
print("Rows with missing target_hedge_ratio_bucket:", n_missing_labels)

# Coverage of all buckets (excluding NaN)
print("\nBucket coverage (non-missing):")
for b in HEDGE_BUCKETS:
    count_b = (df["target_hedge_ratio_bucket"] == b).sum()
    print(f"  bucket {b:.2f}: {count_b} rows")

# --------------------------------------------------------------------
# Usable row count for model training
# --------------------------------------------------------------------
print("\n--- Row Usability for Training ---")

required_for_training = [
    "return_1d", "return_5d", "realized_vol_20d",
    "option_price", "delta", "gamma", "theta", "vega",
    "portfolio_delta", "portfolio_gamma", "portfolio_theta", "portfolio_vega",
    "target_hedge_ratio_bucket",
]

usable = df.dropna(subset=required_for_training)
print("Usable rows for training:", len(usable))
print("Dropped rows due to NaNs:", len(df) - len(usable))

# Optional: show the date range of dropped rows to confirm it's just warm-up/edges
dropped_idx = df.index.difference(usable.index)
if len(dropped_idx) > 0:
    dropped_dates = df.loc[dropped_idx, "date"]
    print("First dropped date:", dropped_dates.min())
    print("Last dropped date:", dropped_dates.max())

# --------------------------------------------------------------------
# Simple P&L sanity snapshot (optional)
# --------------------------------------------------------------------
# If your dataset also includes next-day info or pnl columns, you can
# plug them in here. For now we just check basic distribution stats
# on option_price and close.

print("\n--- Basic Distribution Checks ---")
print("close: min = {:.4f}, max = {:.4f}, mean = {:.4f}".format(
    df["close"].min(), df["close"].max(), df["close"].mean()
))
print("option_price: min = {:.4f}, max = {:.4f}, mean = {:.4f}".format(
    df["option_price"].dropna().min(),
    df["option_price"].dropna().max(),
    df["option_price"].dropna().mean()
))

print("\nSanity check complete.")