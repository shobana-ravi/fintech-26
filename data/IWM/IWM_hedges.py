import pandas as pd
import numpy as np
import argparse


HEDGE_BUCKETS = np.array([0.00, 0.25, 0.50, 0.75, 1.00])


def add_hedge_columns(df):

    for ratio in HEDGE_BUCKETS:
        col_name = f"hedge_{int(ratio*100)}"

        # hedge shares based on portfolio delta
        df[col_name] = -df["portfolio_delta"] * ratio

    return df


def compute_hedges(input_csv, output_csv=None):

    df = pd.read_csv(input_csv)

    if "portfolio_delta" not in df.columns:
        raise ValueError(
            "Missing 'portfolio_delta'. Run portfolio Greeks step first."
        )

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
        default="IWM_data_features_with_options.csv",
        help="IWM dataset with portfolio Greeks"
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output CSV"
    )

    args = parser.parse_args()

    compute_hedges(args.input_csv, args.output)