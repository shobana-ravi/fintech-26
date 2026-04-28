# XGBoost Hedge Model

This folder trains an XGBoost classifier from the final ML CSV and uses the
saved model to predict hedge buckets.

## Expected Input

The training CSV should already be your final dataset and include:

- `close`
- `return_1d`
- `return_5d`
- `realized_vol_20d`
- `strike`
- `T`
- `option_price`
- `delta`
- `gamma`
- `theta`
- `vega`
- `target_hedge_ratio_bucket`

The target bucket must be one of:

- `0.00`
- `0.25`
- `0.50`
- `0.75`
- `1.00`

If the chronological training window does not contain all five buckets, the
trainer automatically remaps the buckets it does see to contiguous class IDs and
saves that mapping with the model.

## Train

From the repo root:

```bash
python3 -m pip install -r requirements.txt
python3 models/train_xgboost.py --input data/DIA/dia_ml_dataset.csv
```

Outputs:

- `models/hedge_xgb.joblib`
- `models/hedge_xgb_metrics.json`

## Predict

Score the latest row of a CSV:

```bash
python3 models/predict_hedge.py --model models/hedge_xgb.joblib --input data/DIA/dia_ml_dataset.csv --latest-only
```

Score a full CSV and save the results:

```bash
python3 models/predict_hedge.py --model models/hedge_xgb.joblib --input data/DIA/dia_ml_dataset.csv --output models/dia_predictions.csv
```
