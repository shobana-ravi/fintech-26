import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_metrics(input_csv: str, output_csv: str | None = None) -> Path:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    # --- Normalize column names (very important fix) ---
    df.columns = [col.lower() for col in df.columns]

    required_cols = {"date", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Clean data ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some date values could not be parsed.")

    df = df.sort_values("date").reset_index(drop=True)

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if df["close"].isna().any():
        raise ValueError("Some close values could not be parsed as numbers.")

    # --- Feature Engineering ---
    df["return_1d"] = df["close"] / df["close"].shift(1) - 1
    df["return_5d"] = df["close"] / df["close"].shift(5) - 1
    df["realized_vol_20d"] = (
        df["return_1d"].rolling(window=20).std() * np.sqrt(252)
    )

    # --- Output path ---
    if output_csv is None:
        output_path = input_path.with_name(f"{input_path.stem}_features.csv")
    else:
        output_path = Path(output_csv)

    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute stock features (returns + volatility)."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="IWM_data.csv",  # <-- updated default
        help="Path to input CSV file (default: iwm_us_d.csv)",
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