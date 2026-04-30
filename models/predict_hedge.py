import argparse
from pathlib import Path

import joblib
import pandas as pd

from xgboost_hedge_features import (
    FEATURE_COLS,
    encoded_prediction_to_bucket,
    engineer_features,
    format_trade_action,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict hedge buckets from a trained XGBoost hedge bundle (XGBoost.py output)."
    )
    parser.add_argument(
        "--model",
        default="outputs/xgboost_hedge_bundle.joblib",
        help="Path to the saved model bundle (xgboost_hedge_bundle.joblib).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV containing rows to score (must include all base feature columns).",
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
    parser.add_argument(
        "--current-hedge-shares",
        type=float,
        default=0.0,
        help="With --latest-only: current hedge position in shares for trade math.",
    )
    return parser.parse_args()


def load_bundle(model_path: Path):
    bundle = joblib.load(model_path)
    required_keys = {"model", "all_features", "label_encoder"}
    missing = required_keys - set(bundle)
    if missing:
        raise ValueError(
            f"Model bundle is missing keys: {sorted(missing)}. "
            "Train with: python models/XGBoost.py --csv data/SPY/spy_training_dataset.csv"
        )
    return bundle


def prepare_base_features(df: pd.DataFrame, feature_cols: list):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required feature columns: " + ", ".join(sorted(missing))
        )
    scored = df.copy()
    scored = scored.dropna(subset=feature_cols).copy()
    if scored.empty:
        raise ValueError("No rows remain after dropping missing base feature values.")
    return scored


def main():
    args = parse_args()

    bundle = load_bundle(Path(args.model))
    model = bundle["model"]
    all_features = list(bundle["all_features"])
    label_encoder = bundle["label_encoder"]
    feature_cols = list(bundle.get("feature_cols", FEATURE_COLS))

    df = pd.read_csv(args.input)
    scored = prepare_base_features(df, feature_cols)

    if args.latest_only:
        scored = scored.tail(1).copy()

    engineered = engineer_features(scored)
    X = engineered[all_features].fillna(0)

    predicted_enc = model.predict(X).astype(int)
    predicted_bucket = [
        float(encoded_prediction_to_bucket(label_encoder, int(v))) for v in predicted_enc
    ]
    probabilities = model.predict_proba(X)
    confidence = probabilities.max(axis=1)

    scored = scored.reset_index(drop=True)
    scored["predicted_hedge_class"] = predicted_enc
    scored["predicted_hedge_ratio_bucket"] = predicted_bucket
    scored["prediction_confidence"] = confidence

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}")

    if args.latest_only:
        row = scored.iloc[-1]
        ratio = float(row["predicted_hedge_ratio_bucket"])
        conf = float(row["prediction_confidence"])
        print(f"Predicted hedge ratio bucket: {ratio:.2f}")
        print(f"Prediction confidence: {conf:.4f}")
        pdelta = float(row["portfolio_delta"])
        target = -pdelta * ratio
        trade = target - float(args.current_hedge_shares)
        print(f"Target hedge shares (delta={pdelta:.2f}, ratio={ratio:.2f}): {target:.2f}")
        print(f"Shares to trade (current hedge={args.current_hedge_shares:.2f}): {trade:.2f}")
        print(f"Action: {format_trade_action(trade)}")
    else:
        print(f"Scored rows: {len(scored)}")
        if args.output is None:
            cols = [
                c
                for c in [
                    "predicted_hedge_ratio_bucket",
                    "prediction_confidence",
                    "portfolio_delta",
                ]
                if c in scored.columns
            ]
            if cols:
                print(scored[cols].tail(10).to_string(index=False))
            else:
                print(scored.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
