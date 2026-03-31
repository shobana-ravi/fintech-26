import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
import argparse

# -----------------------------
# Config
# -----------------------------
HEDGE_BUCKETS = [0.00, 0.25, 0.50, 0.75, 1.00]
COST_PER_SHARE = 0.01  # transaction cost per share
CONTRACT_SIZE = 100     # 1 option contract controls 100 shares

# -----------------------------
# Black-Scholes function
# -----------------------------
def black_scholes_call(S, K, T, r, sigma):
    """
    Returns: price, delta, gamma, theta, vega
    """
    if pd.isna(sigma) or sigma == 0 or T <= 0:
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
def generate_training_dataset(input_csv: str, output_csv: str | None = None):
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    
    df = pd.read_csv(input_path)

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # -----------------------------
    # Standardize column names
    # -----------------------------
    df.rename(columns={'Date':'date', 'Close':'close'}, inplace=True)

    # -----------------------------
    # Step 7: compute next-day option price
    # -----------------------------
    df['option_price_next'] = np.nan
    for t in range(len(df)-1):
        spot_next = df.loc[t+1, 'close']
        K = df.loc[t, 'strike']
        T_next = (df.loc[t, 'dte'] - 1) / 365
        r = df.loc[t, 'r']
        sigma_next = df.loc[t+1, 'realized_vol_20d']
        df.loc[t, 'option_price_next'] = black_scholes_call(spot_next, K, T_next, r, sigma_next)[0]

    # -----------------------------
    # Step 8: option P&L
    # -----------------------------
    df['option_pnl_contract'] = (df['option_price_next'] - df['option_price']) * CONTRACT_SIZE

    # Step 8: compute total P&L for each hedge bucket
    total_pnls_dict = {bucket: [] for bucket in HEDGE_BUCKETS}
    for t in range(len(df)-1):
        spot_today = df.loc[t, 'close']
        spot_next = df.loc[t+1, 'close']
        portfolio_delta = df.loc[t, 'portfolio_delta']
        option_pnl = df.loc[t, 'option_pnl_contract']

        for hedge_ratio in HEDGE_BUCKETS:
            hedge_shares = -portfolio_delta * hedge_ratio
            hedge_pnl = hedge_shares * (spot_next - spot_today)
            hedge_cost = abs(hedge_shares) * COST_PER_SHARE
            total_pnl = option_pnl + hedge_pnl - hedge_cost
            total_pnls_dict[hedge_ratio].append(total_pnl)

    # Last row has no next day
    for bucket in HEDGE_BUCKETS:
        total_pnls_dict[bucket].append(np.nan)

    # -----------------------------
    # Step 9: select best hedge ratio
    # -----------------------------
    target_buckets = []
    for t in range(len(df)):
        total_pnls_today = [total_pnls_dict[bucket][t] for bucket in HEDGE_BUCKETS]
        if all(np.isnan(total_pnls_today)):
            target_buckets.append(np.nan)
        else:
            best_idx = np.nanargmax(total_pnls_today)
            target_buckets.append(HEDGE_BUCKETS[best_idx])
    df['target_hedge_ratio_bucket'] = target_buckets

    # -----------------------------
    # Step 10: build final training dataset
    # -----------------------------
    final_columns = [
        'date', 'close', 'return_1d', 'return_5d', 'realized_vol_20d',
        'strike', 'dte', 'option_price', 'delta', 'gamma', 'theta', 'vega',
        'portfolio_delta', 'portfolio_gamma', 'portfolio_theta', 'portfolio_vega',
        'target_hedge_ratio_bucket'
    ]
    training_df = df[final_columns]

    if output_csv is None:
        output_path = input_path.with_name(f"{input_path.stem}_training_dataset.csv")
    else:
        output_path = Path(output_csv)

    training_df.to_csv(output_path, index=False)
    print(f"Training dataset saved to: {output_path}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training dataset with hedge labels")
    parser.add_argument("input_csv", help="CSV file with portfolio Greeks (from qqq_portfolio_greeks.py)")
    parser.add_argument("-o", "--output", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    generate_training_dataset(args.input_csv, args.output)