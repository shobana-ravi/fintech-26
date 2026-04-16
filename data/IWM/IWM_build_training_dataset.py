import pandas as pd
import argparse


def build_dataset(input_csv, output_csv=None):

    df = pd.read_csv(input_csv)

    print("Detected columns:", df.columns.tolist())


    # --------------------------------------------------
    # Rename columns to match training schema
    # --------------------------------------------------

    rename_map = {
        "spot_today": "close",
        "call_price": "option_price",
        "T": "dte"
    }

    df = df.rename(columns=rename_map)


    # --------------------------------------------------
    # Ensure portfolio Greeks exist
    # --------------------------------------------------

    if "portfolio_delta" not in df.columns:
        df["portfolio_delta"] = df["delta"]

    if "portfolio_gamma" not in df.columns:
        df["portfolio_gamma"] = df["gamma"]

    if "portfolio_theta" not in df.columns:
        df["portfolio_theta"] = df["theta"]

    if "portfolio_vega" not in df.columns:
        df["portfolio_vega"] = df["vega"]


    # --------------------------------------------------
    # Final dataset columns
    # --------------------------------------------------

    final_columns = [

        "date",
        "close",

        "return_1d",
        "return_5d",
        "realized_vol_20d",

        "strike",
        "dte",
        "option_price",

        "delta",
        "gamma",
        "theta",
        "vega",

        "portfolio_delta",
        "portfolio_gamma",
        "portfolio_theta",
        "portfolio_vega",

        "target_hedge_ratio_bucket"
    ]


    # Keep only columns that exist
    final_columns = [c for c in final_columns if c in df.columns]

    df_final = df[final_columns].copy()


    # --------------------------------------------------
    # Save dataset
    # --------------------------------------------------

    if output_csv is None:
        output_csv = "iwm_training_dataset.csv"

    df_final.to_csv(output_csv, index=False)

    print("Training dataset saved:", output_csv)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build IWM hedge ML training dataset"
    )

    parser.add_argument(
        "input_csv",
        help="Input CSV from hedge simulation"
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None
    )

    args = parser.parse_args()

    build_dataset(args.input_csv, args.output)