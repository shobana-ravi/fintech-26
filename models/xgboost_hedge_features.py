"""Shared feature definitions for XGBoost hedge classifier (training + inference)."""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "spot_today",
    "return_1d",
    "return_5d",
    "realized_vol_20d",
    "sigma_next",
    "strike",
    "T",
    "dte_today",
    "call_price",
    "delta",
    "gamma",
    "theta",
    "vega",
    "portfolio_delta",
    "portfolio_gamma",
    "portfolio_theta",
    "portfolio_vega",
    "option_pnl",
]

TARGET_COL = "target_class"

# Live dashboards often send option_pnl=0; training labels use a one-day PnL proxy
# with a non-zero distribution. Imputing the SPY training median avoids spurious 0% hedge.
SPY_TRAINING_MEDIAN_OPTION_PNL = -0.0420985126464898

# Same semantics as data/SPY/build_final_training_set.py ratio_to_class (inverse: class -> bucket)
CLASS_INDEX_TO_BUCKET = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["moneyness"] = df["spot_today"] / df["strike"]
    df["vol_spread"] = df["sigma_next"] - df["realized_vol_20d"]
    df["delta_gamma_ratio"] = df["delta"] / (df["gamma"].replace(0, np.nan))
    df["theta_vega_ratio"] = df["theta"] / (df["vega"].replace(0, np.nan))
    df["port_delta_norm"] = df["portfolio_delta"] / df["spot_today"]
    return df


ENGINEERED = [
    "moneyness",
    "vol_spread",
    "delta_gamma_ratio",
    "theta_vega_ratio",
    "port_delta_norm",
]

ALL_FEATURES = FEATURE_COLS + ENGINEERED


def encoded_prediction_to_bucket(label_encoder, predicted_enc: int) -> float:
    """Map XGBoost class index (0..n-1) to hedge ratio bucket using training label semantics."""
    orig = int(float(label_encoder.classes_[int(predicted_enc)]))
    if orig not in CLASS_INDEX_TO_BUCKET:
        raise ValueError(f"Unexpected target_class label: {orig}")
    return CLASS_INDEX_TO_BUCKET[orig]


def format_trade_action(shares_to_trade: float) -> str:
    if shares_to_trade == 0 or (isinstance(shares_to_trade, float) and np.isnan(shares_to_trade)):
        return "no trade"
    n = abs(int(round(shares_to_trade)))
    if shares_to_trade > 0:
        return f"buy {n} shares"
    return f"short {n} shares"
