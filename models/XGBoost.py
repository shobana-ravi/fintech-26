import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from xgboost_hedge_features import (
    ALL_FEATURES,
    CLASS_INDEX_TO_BUCKET,
    ENGINEERED,
    FEATURE_COLS,
    TARGET_COL,
    encoded_prediction_to_bucket,
    engineer_features,
)

warnings.filterwarnings("ignore")


def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df):,} rows")

    missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    return df


def build_model(n_classes):
    return XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )


def make_class_weights(y_train):
    counts = pd.Series(y_train).value_counts().sort_index()
    total = counts.sum()
    n_classes = len(counts)

    class_weight_map = {
        cls: total / (n_classes * count)
        for cls, count in counts.items()
    }

    sample_weight = np.array([class_weight_map[int(y)] for y in y_train], dtype=float)

    print("\nClass counts:")
    print(counts.to_string())

    print("\nClass weights:")
    for cls, wt in class_weight_map.items():
        print(f"class {cls}: {wt:.4f}")

    return sample_weight, class_weight_map


def make_plots(model, X_test, y_test, le, feature_names, out_path, title_prefix):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    y_pred = model.predict(X_test)

    present_classes = sorted(set(y_test) | set(y_pred))
    label_map = {
        cls: f"{int(encoded_prediction_to_bucket(le, cls) * 100)}%"
        for cls in present_classes
    }

    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[label_map[c] for c in present_classes],
    )
    disp.plot(ax=ax1, colorbar=False, cmap="Blues")
    ax1.set_title("Confusion Matrix", color="white")
    ax1.set_facecolor("#1a1d27")
    ax1.tick_params(colors="white")

    ax2 = fig.add_subplot(gs[0, 1])
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    top = imp.tail(15)
    ax2.barh(top.index, top.values)
    ax2.set_title("Top Feature Importances", color="white")
    ax2.set_facecolor("#1a1d27")
    ax2.tick_params(colors="white")

    ax3 = fig.add_subplot(gs[1, 0])
    pd.Series(y_test).value_counts().sort_index().plot(kind="bar", ax=ax3)
    ax3.set_title("Target Class Distribution", color="white")
    ax3.set_facecolor("#1a1d27")
    ax3.tick_params(colors="white")

    ax4 = fig.add_subplot(gs[1, 1])
    proba = model.predict_proba(X_test)
    im = ax4.imshow(proba, aspect="auto")
    fig.colorbar(im, ax=ax4)
    ax4.set_title("Prediction Probabilities", color="white")
    ax4.set_facecolor("#1a1d27")
    ax4.tick_params(colors="white")

    fig.suptitle(f"{title_prefix} XGBoost Hedge Ratio Classifier", color="white")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"Plot saved -> {out_path}")


def main(csv_path, ticker):
    ticker = ticker.upper()

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

    print(f"Ticker: {ticker}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    sample_weight, class_weight_map = make_class_weights(y_train)

    model = build_model(n_classes)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, f"{ticker.lower()}_xgboost_hedge_results.png")
    make_plots(model, X_test, y_test, le, ALL_FEATURES, plot_path, ticker)

    model_path = os.path.join(output_dir, f"{ticker.lower()}_xgboost_hedge_model.json")
    model.save_model(model_path)
    print(f"Model saved -> {model_path}")

    bundle_path = os.path.join(output_dir, f"{ticker.lower()}_xgboost_hedge_bundle.joblib")
    joblib.dump(
        {
            "model": model,
            "all_features": ALL_FEATURES,
            "class_index_to_bucket": CLASS_INDEX_TO_BUCKET,
            "label_encoder": le,
            "feature_cols": FEATURE_COLS,
            "engineered_cols": ENGINEERED,
            "class_weight_map": class_weight_map,
        },
        bundle_path,
    )
    print(f"Bundle saved -> {bundle_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. SPY")
    args = parser.parse_args()

    main(args.csv, args.ticker)