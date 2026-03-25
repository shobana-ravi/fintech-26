import pandas as pd
import numpy as np
import argparse


HEDGE_BUCKETS = np.array([0.00, 0.25, 0.50, 0.75, 1.00])


def add_hedge_columns(df):
    for ratio in HEDGE_BUCKETS:
        col_name = f"hedge_{int(ratio*100)}"
        df[col_name] = -df["delta"] * ratio
    return df


def compute_hedges(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)

    if "delta" not in df.columns:
        raise ValueError("Missing 'delta' column. Run Black-Scholes step first.")

    df = add_hedge_columns(df)

    if output_csv is None:
        output_csv = input_csv.replace(".csv", "_with_hedges.csv")

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