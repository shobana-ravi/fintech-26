"""
XGBoost Hedge Ratio Classifier — IWM Options Dataset
======================================================
Usage:
  python XGBoost.py --csv iwm_ml_dataset_fixed.csv
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# FEATURE COLUMNS
# ──────────────────────────────────────────────
FEATURE_COLS = [
    "spot_today", "return_1d", "return_5d",
    "realized_vol_20d", "sigma_next",
    "strike", "T", "dte_today",
    "call_price", "delta", "gamma", "theta", "vega",
    "portfolio_delta", "portfolio_gamma", "portfolio_theta", "portfolio_vega",
    "option_pnl",
]
TARGET_COL = "target_class"

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df):,} rows")

    # Validate required columns
    missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    return df

# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["moneyness"]         = df["spot_today"] / df["strike"]
    df["vol_spread"]        = df["sigma_next"] - df["realized_vol_20d"]
    df["delta_gamma_ratio"] = df["delta"] / (df["gamma"].replace(0, np.nan))
    df["theta_vega_ratio"]  = df["theta"] / (df["vega"].replace(0, np.nan))
    df["port_delta_norm"]   = df["portfolio_delta"] / df["spot_today"]
    return df

ENGINEERED = [
    "moneyness", "vol_spread",
    "delta_gamma_ratio", "theta_vega_ratio",
    "port_delta_norm"
]

ALL_FEATURES = FEATURE_COLS + ENGINEERED

# ──────────────────────────────────────────────
# BUILD MODEL
# ──────────────────────────────────────────────
def build_model(n_classes):
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softmax",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )

# ──────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────
def make_plots(model, X_test, y_test, le, feature_names, out_path):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    label_map = {i: f"{int(le.classes_[i]*100)}%" for i in range(len(le.classes_))}

    # Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[label_map[c] for c in range(len(le.classes_))]
    )
    disp.plot(ax=ax1, colorbar=False, cmap="Blues")
    ax1.set_title("Confusion Matrix", color="white")
    ax1.set_facecolor("#1a1d27")
    ax1.tick_params(colors="white")

    # Feature Importance
    ax2 = fig.add_subplot(gs[0, 1])
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    top = imp.tail(15)
    ax2.barh(top.index, top.values)
    ax2.set_title("Top Feature Importances", color="white")
    ax2.set_facecolor("#1a1d27")
    ax2.tick_params(colors="white")

    # Class Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    pd.Series(y_test).value_counts().sort_index().plot(kind="bar", ax=ax3)
    ax3.set_title("Target Class Distribution", color="white")
    ax3.set_facecolor("#1a1d27")
    ax3.tick_params(colors="white")

    # Probability Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    proba = model.predict_proba(X_test)
    im = ax4.imshow(proba, aspect="auto")
    fig.colorbar(im, ax=ax4)

    fig.suptitle("XGBoost Hedge Ratio Classifier", color="white")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"Plot saved → {out_path}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main(csv_path):
    df = load_data(csv_path)
    df = engineer_features(df)

    le = LabelEncoder()
    df["y"] = le.fit_transform(df[TARGET_COL])
    n_classes = len(le.classes_)

    X = df[ALL_FEATURES].fillna(0)
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    model = build_model(n_classes)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create outputs folder
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save plots
    plot_path = os.path.join(output_dir, "xgboost_hedge_results.png")
    make_plots(model, X_test, y_test, le, ALL_FEATURES, plot_path)

    # Save model
    model_path = os.path.join(output_dir, "xgboost_hedge_model.json")
    model.save_model(model_path)
    print(f"Model saved → {model_path}")

# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    args = parser.parse_args()

    main(args.csv)