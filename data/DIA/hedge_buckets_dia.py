import pandas as pd
import numpy as np
import argparse

HEDGE_BUCKETS = np.array([0.00, 0.25, 0.50, 0.75, 1.00])

RATIO_TO_CLASS = {
    0.00: 0,
    0.25: 1,
    0.50: 2,
    0.75: 3,
    1.00: 4,
}


def add_hedge_columns(df):
    # Assume df["delta"] is option delta per share for 1 option
    # Convert to portfolio delta for 1 contract
    df["portfolio_delta"] = df["delta"] * 100

    for ratio in HEDGE_BUCKETS:
        col_name = f"hedge_{int(ratio * 100)}"
        df[col_name] = -df["portfolio_delta"] * ratio

    return df


def add_target_labels(df, tx_cost_per_share=0.01):
    required = ["close", "option_price", "portfolio_delta"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Next-day values
    df["close_next"] = df["close"].shift(-1)
    df["option_price_next"] = df["option_price"].shift(-1)

    # Long 1 contract => multiply option price change by 100
    df["option_pnl"] = (df["option_price_next"] - df["option_price"]) * 100

    best_ratios = []
    best_classes = []

    for i, row in df.iterrows():
        if pd.isna(row["close_next"]) or pd.isna(row["option_price_next"]):
            best_ratios.append(np.nan)
            best_classes.append(np.nan)
            continue

        best_ratio = None
        best_total_pnl = -np.inf

        for ratio in HEDGE_BUCKETS:
            hedge_shares = -row["portfolio_delta"] * ratio
            hedge_pnl = hedge_shares * (row["close_next"] - row["close"])
            hedge_cost = abs(hedge_shares) * tx_cost_per_share

            total_pnl = row["option_pnl"] + hedge_pnl - hedge_cost

            if total_pnl > best_total_pnl:
                best_total_pnl = total_pnl
                best_ratio = ratio

        best_ratios.append(best_ratio)
        best_classes.append(RATIO_TO_CLASS[best_ratio])

    df["target_hedge_ratio_bucket"] = best_ratios
    df["target_class"] = best_classes

    return df


def compute_hedges(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)

    if "delta" not in df.columns:
        raise ValueError("Missing 'delta' column. Run Black-Scholes step first.")

    if "option_price" not in df.columns:
        raise ValueError("Missing 'option_price' column. Need option price to compute labels.")

    if "close" not in df.columns:
        raise ValueError("Missing 'close' column. Need stock price to compute hedge P&L.")

    df = add_hedge_columns(df)
    df = add_target_labels(df)

    if output_csv is None:
        output_csv = input_csv.replace(".csv", "_with_hedges_and_labels.csv")

    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="dia_us_d_with_metrics_with_options.csv",
    )
    parser.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    compute_hedges(args.input_csv, args.output)