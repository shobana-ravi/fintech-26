import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


BUCKETS = [0.00, 0.25, 0.50, 0.75, 1.00]
DEFAULT_FEATURES = [
    "close",
    "return_1d",
    "return_5d",
    "realized_vol_20d",
    "strike",
    "T",
    "option_price",
    "delta",
    "gamma",
    "theta",
    "vega",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost hedge-ratio model from the final ML CSV."
    )
    parser.add_argument(
        "--input",
        default="data/DIA/dia_ml_dataset.csv",
        help="Path to the final training CSV.",
    )
    parser.add_argument(
        "--output-model",
        default="models/hedge_xgb.joblib",
        help="Where to save the trained model bundle.",
    )
    parser.add_argument(
        "--output-metrics",
        default="models/hedge_xgb_metrics.json",
        help="Where to save evaluation metrics.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Fraction of rows used for the chronological training split.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of rows used for the chronological validation split.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Fraction of rows used for the chronological test split.",
    )
    return parser.parse_args()


def normalize_target(value):
    value = round(float(value), 2)
    if value not in BUCKETS:
        raise ValueError(
            f"Unexpected hedge bucket {value}. Expected one of {sorted(BUCKETS)}."
        )
    return value


def load_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)

    if "target_hedge_ratio_bucket" not in df.columns:
        raise ValueError("CSV must contain 'target_hedge_ratio_bucket'.")

    missing_features = [col for col in DEFAULT_FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(
            "CSV is missing required feature columns: "
            + ", ".join(sorted(missing_features))
        )

    required_columns = DEFAULT_FEATURES + ["target_hedge_ratio_bucket"]
    clean_df = df[required_columns].dropna().copy()
    clean_df["target_bucket"] = clean_df["target_hedge_ratio_bucket"].apply(
        normalize_target
    )
    return clean_df


def split_dataset(
    df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float
):
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train-frac + val-frac + test-frac must sum to 1.0")

    n_rows = len(df)
    if n_rows < 20:
        raise ValueError(
            "Need at least 20 usable rows to train and evaluate the model."
        )

    train_end = int(n_rows * train_frac)
    val_end = train_end + int(n_rows * val_frac)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "One of the chronological splits is empty. Adjust the split fractions."
        )

    return train_df, val_df, test_df


def build_class_mapping(train_df: pd.DataFrame):
    observed_buckets = sorted(train_df["target_bucket"].unique())
    bucket_to_class = {
        float(bucket): idx for idx, bucket in enumerate(observed_buckets)
    }
    class_to_bucket = {idx: bucket for bucket, idx in bucket_to_class.items()}
    return bucket_to_class, class_to_bucket


def encode_targets(df: pd.DataFrame, bucket_to_class):
    encoded = df.copy()
    encoded["target_class"] = encoded["target_bucket"].map(bucket_to_class)
    return encoded


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, num_classes: int):
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        eval_metric="mlogloss",
        random_state=42,
    )

    model.fit(
        train_df[DEFAULT_FEATURES],
        train_df["target_class"],
        eval_set=[
            (train_df[DEFAULT_FEATURES], train_df["target_class"]),
            (val_df[DEFAULT_FEATURES], val_df["target_class"]),
        ],
        verbose=False,
    )
    return model


def evaluate_model(model: XGBClassifier, test_df: pd.DataFrame):
    eval_df = test_df.dropna(subset=["target_class"]).copy()
    skipped_rows = int(len(test_df) - len(eval_df))

    if eval_df.empty:
        return {
            "accuracy": None,
            "confusion_matrix": [],
            "classification_report": {},
            "test_rows": int(len(test_df)),
            "evaluated_rows": 0,
            "skipped_unseen_bucket_rows": skipped_rows,
        }

    y_true = eval_df["target_class"].astype(int)
    y_pred = model.predict(eval_df[DEFAULT_FEATURES])

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
        "test_rows": int(len(test_df)),
        "evaluated_rows": int(len(eval_df)),
        "skipped_unseen_bucket_rows": skipped_rows,
    }
    return metrics


def save_outputs(model, metrics, output_model: Path, output_metrics: Path):
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": DEFAULT_FEATURES,
            "class_to_bucket": metrics["class_to_bucket"],
            "bucket_to_class": metrics["bucket_to_class"],
        },
        output_model,
    )

    with output_metrics.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_model = Path(args.output_model)
    output_metrics = Path(args.output_metrics)

    dataset = load_dataset(input_path)
    train_df, val_df, test_df = split_dataset(
        dataset, args.train_frac, args.val_frac, args.test_frac
    )
    bucket_to_class, class_to_bucket = build_class_mapping(train_df)
    train_df = encode_targets(train_df, bucket_to_class)
    val_df = encode_targets(val_df, bucket_to_class)
    test_df = encode_targets(test_df, bucket_to_class)

    model = train_model(train_df, val_df, len(bucket_to_class))
    metrics = evaluate_model(model, test_df)
    metrics["bucket_to_class"] = bucket_to_class
    metrics["class_to_bucket"] = class_to_bucket
    metrics["observed_training_buckets"] = sorted(bucket_to_class)
    save_outputs(model, metrics, output_model, output_metrics)

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Saved model to: {output_model}")
    print(f"Saved metrics to: {output_metrics}")
    print(f"Observed training buckets: {metrics['observed_training_buckets']}")
    if metrics["accuracy"] is None:
        print(
            "Test accuracy: unavailable because no test rows matched the training buckets"
        )
    else:
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        if metrics["skipped_unseen_bucket_rows"]:
            print(
                "Skipped unseen-bucket test rows: "
                f"{metrics['skipped_unseen_bucket_rows']}"
            )


if __name__ == "__main__":
    main()
