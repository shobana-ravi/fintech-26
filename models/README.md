# XGBoost Hedge Model

This folder contains two training paths:

1. **`XGBoost.py` (dashboard / production inference)** — full feature set with engineered columns; saves `outputs/xgboost_hedge_bundle.joblib` for the API and `outputs/xgboost_hedge_model.json` for portability.
2. **`train_xgboost.py` (legacy DIA-style CSV)** — simpler 12-column schema and `models/hedge_xgb.joblib` (not used by the hedge dashboard refresh flow).

## Production bundle (refresh recommendation)

Train on the final SPY dataset whose columns match `models/xgboost_hedge_features.py` (same schema as `data/SPY/build_final_training_set.py` output):

```bash
python3 -m pip install -r requirements.txt
python3 models/XGBoost.py --csv data/SPY/spy_training_dataset.csv
```

Outputs:

- `outputs/xgboost_hedge_bundle.joblib` — **used by** `backend/api_server.py` for `/api/hedge/recommend`
- `outputs/xgboost_hedge_model.json` — native XGBoost JSON
- `outputs/xgboost_hedge_results.png`

Start the API from the repo root (default port **8001** to match the Vite frontend):

```bash
python3 backend/api_server.py
```

The recommend endpoint expects JSON with every **base** feature in `FEATURE_COLS` (see `models/xgboost_hedge_features.py`), plus:

- `portfolio_delta` — **share-equivalent** total delta (required for target hedge and trade math)
- `current_hedge_shares` — optional, defaults to `0`

The API returns `predicted_hedge_ratio_bucket`, `target_hedge_shares`, `shares_to_trade`, and `action`, among other fields.

## Predict (CLI)

Score the latest row of a CSV (must include all base feature columns):

```bash
python3 models/predict_hedge.py --model outputs/xgboost_hedge_bundle.joblib --input data/SPY/spy_training_dataset.csv --latest-only --current-hedge-shares -20
```

## Legacy `train_xgboost.py` (12-column CSV)

For older DIA-style datasets with `close`, `option_price`, and `target_hedge_ratio_bucket`:

```bash
python3 models/train_xgboost.py --input data/DIA/dia_ml_dataset.csv
```

Outputs: `models/hedge_xgb.joblib`, `models/hedge_xgb_metrics.json`.
