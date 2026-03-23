import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_metrics(input_csv: str, output_csv: str | None = None) -> Path:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    # --- CLEAN COLUMN NAMES ---
    df.columns = df.columns.str.strip()

    # Handle case-insensitive column names
    col_map = {col.lower(): col for col in df.columns}

    if "date" not in col_map or "close" not in col_map:
        raise ValueError("CSV must contain Date and Close columns.")

    date_col = col_map["date"]
    close_col = col_map["close"]

    # --- PARSE DATA ---
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=[date_col, close_col])

    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)

    # --- METRICS ---
    df["return_1d"] = df[close_col] / df[close_col].shift(1) - 1
    df["return_5d"] = df[close_col] / df[close_col].shift(5) - 1
    df["realized_vol_20d"] = df["return_1d"].rolling(window=20).std() * np.sqrt(252)

    # --- OUTPUT ---
    if output_csv is None:
        output_path = input_path.with_name(f"{input_path.stem}_with_metrics.csv")
    else:
        output_path = Path(output_csv)

    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute return_1d, return_5d, and realized_vol_20d from a price CSV."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="dia_us_d.csv",
        help="Path to input CSV file (default: dia_us_d.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output CSV path. Default adds _with_metrics suffix.",
    )
    args = parser.parse_args()

    output_file = compute_metrics(args.input_csv, args.output)
    print(f"Saved output to: {output_file}")
