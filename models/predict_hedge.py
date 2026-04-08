import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict hedge buckets from a trained XGBoost model."
    )
    parser.add_argument(
        "--model",
        default="models/hedge_xgb.joblib",
        help="Path to the saved model bundle.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV containing rows to score.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path for scored predictions.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Score only the last row of the input CSV and print the recommendation.",
    )
    return parser.parse_args()


def load_bundle(model_path: Path):
    bundle = joblib.load(model_path)
    required_keys = {"model", "features", "class_to_bucket"}
    missing = required_keys - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    return bundle


def prepare_features(df: pd.DataFrame, features):
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required feature columns: "
            + ", ".join(sorted(missing))
        )

    scored = df.copy()
    scored = scored.dropna(subset=features).copy()
    if scored.empty:
        raise ValueError("No rows remain after dropping missing feature values.")
    return scored


def main():
    args = parse_args()

    bundle = load_bundle(Path(args.model))
    model = bundle["model"]
    features = bundle["features"]
    class_to_bucket = bundle["class_to_bucket"]

    df = pd.read_csv(args.input)
    scored = prepare_features(df, features)

    if args.latest_only:
        scored = scored.tail(1).copy()

    predicted_class = model.predict(scored[features])
    predicted_bucket = [class_to_bucket[int(value)] for value in predicted_class]

    probabilities = model.predict_proba(scored[features])
    confidence = probabilities.max(axis=1)

    scored["predicted_hedge_class"] = predicted_class.astype(int)
    scored["predicted_hedge_ratio_bucket"] = predicted_bucket
    scored["prediction_confidence"] = confidence

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}")

    if args.latest_only:
        row = scored.iloc[-1]
        print(f"Predicted hedge ratio bucket: {row['predicted_hedge_ratio_bucket']:.2f}")
        print(f"Prediction confidence: {row['prediction_confidence']:.4f}")
    else:
        print(f"Scored rows: {len(scored)}")
        if args.output is None:
            print(scored.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
