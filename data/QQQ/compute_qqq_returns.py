import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RISK_FREE_RATE = 0.03  # constant


def compute_metrics(input_csv: str, output_csv: str | None = None) -> Path:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = {"Date", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Some Date values could not be parsed.")

    df = df.sort_values("Date").reset_index(drop=True)

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if df["Close"].isna().any():
        raise ValueError("Some Close values could not be parsed as numbers.")

    # --- RETURNS + VOL ---
    df["return_1d"] = df["Close"] / df["Close"].shift(1) - 1
    df["return_5d"] = df["Close"] / df["Close"].shift(5) - 1
    df["realized_vol_20d"] = df["return_1d"].rolling(window=20).std() * np.sqrt(252)

    # --- SYNTHETIC OPTION INPUTS ---
    df["spot"] = df["Close"]

    # nearest ATM strike (MVP)
    df["strike"] = df["spot"].round()

    df["dte"] = 30
    df["T"] = 30 / 365

    # volatility input
    df["sigma"] = df["realized_vol_20d"]

    df["r"] = RISK_FREE_RATE
    df["option_type"] = "call"

    # Optional: reorder columns nicely
    cols = [
        "Date",
        "Close",
        "spot",
        "strike",
        "dte",
        "T",
        "sigma",
        "r",
        "option_type",
        "return_1d",
        "return_5d",
        "realized_vol_20d",
    ]
    df = df[cols]

    if output_csv is None:
        output_path = input_path.with_name(f"{input_path.stem}_with_metrics.csv")
    else:
        output_path = Path(output_csv)

    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute returns, realized vol, and synthetic option inputs."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="qqq_us_d_with_metrics.csv",
        help="Path to input CSV file (default: qqq_us_d.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output CSV path.",
    )
    args = parser.parse_args()

    output_file = compute_metrics(args.input_csv, args.output)
    print(f"Saved output to: {output_file}")