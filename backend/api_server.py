import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.bs_utils import apply_training_style_option_row
from models.xgboost_hedge_features import (
    FEATURE_COLS,
    SPY_TRAINING_MEDIAN_OPTION_PNL,
    encoded_prediction_to_bucket,
    engineer_features,
    format_trade_action,
)

MODEL_PATHS = {
    "SPY": _REPO_ROOT / "outputs" / "spy_xgboost_hedge_bundle.joblib",
    "QQQ": _REPO_ROOT / "outputs" / "qqq_xgboost_hedge_bundle.joblib",
    "DIA": _REPO_ROOT / "outputs" / "dia_xgboost_hedge_bundle.joblib",
    "IWM": _REPO_ROOT / "outputs" / "iwm_xgboost_hedge_bundle.joblib",
}

LATEST_FEATURE_ROW_PATHS = {
    "SPY": _REPO_ROOT / "data" / "SPY" / "spy_training_dataset.csv",
    "QQQ": _REPO_ROOT / "data" / "QQQ" / "qqq_training_dataset_fixed.csv",
    "DIA": _REPO_ROOT / "data" / "DIA" / "dia_ml_dataset_fixed.csv",
    "IWM": _REPO_ROOT / "data" / "IWM" / "iwm_ml_dataset_fixed.csv",
}

SPY_TRAINING_STATS_CSV = _REPO_ROOT / "data" / "SPY" / "spy_training_dataset.csv"
DIA_CSV_PATH = _REPO_ROOT / "data" / "DIA" / "dia_us_d.csv"
IWM_CSV_PATH = _REPO_ROOT / "data" / "IWM" / "IWM_data.csv"
QQQ_CSV_PATH = _REPO_ROOT / "data" / "QQQ" / "qqq_us_d.csv"
SPY_CSV_PATH = _REPO_ROOT / "data" / "SPY" / "spy_us_d.csv"
PAPER_ORDERS_PATH = _REPO_ROOT / "data" / "paper_orders.json"
DEBUG_LOG_PATH = _REPO_ROOT / "outputs" / "agent_debug.log"

HOST = "0.0.0.0"
PORT = 8001

SUPPORTED_TICKERS = {"SPY", "QQQ", "DIA", "IWM"}

TICKER_CSV_MAP = {
    "SPY": SPY_CSV_PATH,
    "QQQ": QQQ_CSV_PATH,
    "DIA": DIA_CSV_PATH,
    "IWM": IWM_CSV_PATH,
}


def debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": "hedge-api",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(pd.Timestamp.now("UTC").timestamp() * 1000),
    }
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8", errors="replace", newline="\n") as handle:
            handle.write(json.dumps(payload) + "\n")
    except OSError:
        pass


def _compute_annualized_vol(csv_path: Path) -> float:
    if not csv_path.exists():
        return 0.15

    try:
        df = pd.read_csv(csv_path)

        if "git aDate" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"git aDate": "Date"})

        if "Close" in df.columns:
            prices = pd.to_numeric(df["Close"], errors="coerce").dropna()
        elif "Last Price" in df.columns:
            prices = pd.to_numeric(df["Last Price"], errors="coerce").dropna()
        else:
            return 0.15

        if len(prices) < 10:
            return 0.15

        log_rets = np.log(prices.values[1:] / prices.values[:-1])
        rv = float(np.std(log_rets, ddof=1)) * np.sqrt(252)
        return rv if np.isfinite(rv) and rv > 1e-6 else 0.15
    except Exception:
        return 0.15


def _to_numeric_percent(value):
    if pd.isna(value):
        return None
    cleaned = str(value).replace("%", "").strip()
    if cleaned in {"", "-"}:
        return None
    return float(cleaned)


def load_latest_feature_row(ticker: str) -> dict:
    ticker = ticker.upper()
    if ticker not in LATEST_FEATURE_ROW_PATHS:
        raise ValueError(f"Unsupported ticker: {ticker}")

    csv_path = LATEST_FEATURE_ROW_PATHS[ticker]
    if not csv_path.exists():
        raise FileNotFoundError(f"Latest feature dataset not found for {ticker}: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Latest feature dataset is empty for {ticker}")

    df.columns = [c.strip() for c in df.columns]
    row = df.iloc[-1].to_dict()

    if "close" in row and "spot_today" not in row:
        row["spot_today"] = row["close"]
    if "option_price" in row and "call_price" not in row:
        row["call_price"] = row["option_price"]
    if "dte" in row and "dte_today" not in row:
        row["dte_today"] = row["dte"]
    if "T" in row and "dte_today" not in row:
        try:
            row["dte_today"] = round(float(row["T"]) * 365)
        except Exception:
            pass
    if "realized_vol_20d" in row and "sigma_next" not in row:
        row["sigma_next"] = row["realized_vol_20d"]

    if "option_pnl" not in row:
        row["option_pnl"] = 0.0

    # Important: do NOT trust dataset portfolio_delta for live sizing.
    # We recompute it from option delta * 100 * contract_count.
    row["ticker"] = ticker
    return row


class ModelService:
    def __init__(self, model_path: Path, ticker: str):
        if not model_path.exists():
            raise FileNotFoundError(f"Model bundle not found for {ticker}: {model_path}")

        self.ticker = ticker
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.all_features = list(self.bundle["all_features"])
        self.label_encoder = self.bundle["label_encoder"]
        self.feature_cols = list(self.bundle.get("feature_cols", FEATURE_COLS))
        self._return_1d_clip = (-float("inf"), float("inf"))
        self._return_5d_clip = (-float("inf"), float("inf"))

        if SPY_TRAINING_STATS_CSV.exists():
            try:
                stats = pd.read_csv(SPY_TRAINING_STATS_CSV, usecols=["return_1d", "return_5d"])
                qlo, qhi = 0.01, 0.99
                self._return_1d_clip = (
                    float(stats["return_1d"].quantile(qlo)),
                    float(stats["return_1d"].quantile(qhi)),
                )
                self._return_5d_clip = (
                    float(stats["return_5d"].quantile(qlo)),
                    float(stats["return_5d"].quantile(qhi)),
                )
            except (ValueError, OSError, KeyError):
                pass

        self._ticker_vol = {t: _compute_annualized_vol(path) for t, path in TICKER_CSV_MAP.items()}
        self._spy_vol = self._ticker_vol.get("SPY", 0.15) or 0.15

    def _clip_returns_like_training(self, row: dict) -> None:
        lo, hi = self._return_1d_clip
        row["return_1d"] = min(max(float(row["return_1d"]), lo), hi)
        lo, hi = self._return_5d_clip
        row["return_5d"] = min(max(float(row["return_5d"]), lo), hi)

    def predict(self, payload: dict) -> dict:
        missing = [c for c in self.feature_cols if c not in payload]
        if missing:
            raise ValueError(f"Missing required feature fields: {', '.join(missing)}")
        if "portfolio_delta" not in payload:
            raise ValueError("Missing required field: portfolio_delta")

        ticker = str(payload.get("ticker", self.ticker)).upper()
        user_portfolio_delta = float(payload["portfolio_delta"])
        current_hedge = float(payload.get("current_hedge_shares", 0) or 0)

        row = {c: float(payload[c]) for c in self.feature_cols}

        payload_vol = float(payload.get("realized_vol_20d", 0.0))
        startup_vol = self._ticker_vol.get(ticker, self._spy_vol)
        ticker_vol = payload_vol if payload_vol > 1e-6 else startup_vol
        vol_ratio = ticker_vol / self._spy_vol if self._spy_vol > 1e-6 else 1.0

        row["realized_vol_20d"] = ticker_vol
        row["sigma_next"] = float(payload.get("sigma_next", min(ticker_vol * 1.02 + 0.001, 2.5)))

        apply_training_style_option_row(row)

        row["realized_vol_20d"] = ticker_vol
        row["sigma_next"] = min(ticker_vol * 1.02 + 0.001, 2.5)

        for greek in ("vega", "gamma", "theta"):
            if greek in row:
                row[greek] = row[greek] * vol_ratio

        # Important: portfolio greeks should scale with contract count, not fixed 1 contract
        contract_count = float(payload.get("contract_count", 1) or 1)
        row["portfolio_delta"] = float(row["delta"]) * 100.0 * contract_count
        if "gamma" in row:
            row["portfolio_gamma"] = float(row["gamma"]) * 100.0 * contract_count
        if "theta" in row:
            row["portfolio_theta"] = float(row["theta"]) * 100.0 * contract_count
        if "vega" in row:
            row["portfolio_vega"] = float(row["vega"]) * 100.0 * contract_count

        if vol_ratio > 1e-6:
            row["return_1d"] = float(row["return_1d"]) / vol_ratio
            row["return_5d"] = float(row["return_5d"]) / vol_ratio

        if abs(float(row.get("option_pnl", 0.0))) < 1e-8:
            row["option_pnl"] = SPY_TRAINING_MEDIAN_OPTION_PNL

        self._clip_returns_like_training(row)

        frame = pd.DataFrame([row])
        frame = engineer_features(frame)
        scored = frame[self.all_features].fillna(0)

        predicted_enc = int(self.model.predict(scored)[0])
        probabilities = np.asarray(self.model.predict_proba(scored)[0], dtype=float)
        confidence = float(probabilities.max())
        predicted_bucket = float(encoded_prediction_to_bucket(self.label_encoder, predicted_enc))

        target_hedge_shares = -user_portfolio_delta * predicted_bucket
        shares_to_trade = target_hedge_shares - current_hedge
        action = format_trade_action(shares_to_trade)

        return {
            "predicted_hedge_class": predicted_enc,
            "predicted_hedge_ratio_bucket": predicted_bucket,
            "prediction_confidence": confidence,
            "features_used": self.all_features,
            "base_features": self.feature_cols,
            "ticker": ticker,
            "model_used": self.ticker,
            "ticker_vol": round(ticker_vol, 6),
            "vol_ratio": round(vol_ratio, 6),
            "contract_count": contract_count,
            "portfolio_delta": user_portfolio_delta,
            "current_hedge_shares": current_hedge,
            "target_hedge_shares": target_hedge_shares,
            "shares_to_trade": shares_to_trade,
            "action": action,
        }


def load_model_services():
    services = {}
    errors = {}

    for ticker, path in MODEL_PATHS.items():
        try:
            services[ticker] = ModelService(path, ticker)
        except Exception as err:
            errors[ticker] = str(err)

    if not services:
        details = "; ".join(f"{t}: {e}" for t, e in errors.items())
        raise RuntimeError(f"No model bundles could be loaded. {details}")

    return services, errors


MODEL_SERVICES, MODEL_LOAD_ERRORS = load_model_services()


class QuoteService:
    def __init__(self, csv_paths):
        self.csv_paths = csv_paths
        self.quotes = {}
        self.frames = {}
        self.errors = {}
        self._load_all()

    def _load_all(self):
        for ticker, csv_path in self.csv_paths.items():
            try:
                quote, frame = self._load_quote(csv_path, ticker)
                self.quotes[ticker] = quote
                self.frames[ticker] = frame
            except Exception as err:
                self.errors[ticker] = str(err)

    def _load_quote(self, csv_path: Path, ticker: str):
        if not csv_path.exists():
            raise FileNotFoundError(f"{ticker} CSV not found at {csv_path}")

        frame = pd.read_csv(csv_path)
        if "git aDate" in frame.columns and "Date" not in frame.columns:
            frame = frame.rename(columns={"git aDate": "Date"})

        columns = set(frame.columns)

        if {"Date", "Close"}.issubset(columns):
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
            frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")

            if len(frame) < 2:
                raise ValueError(f"{ticker} CSV must contain at least two valid rows")

            previous_close = float(frame.iloc[-2]["Close"])
            latest_close = float(frame.iloc[-1]["Close"])
            change_pct = ((latest_close / previous_close) - 1.0) * 100.0
            as_of = frame.iloc[-1]["Date"].date().isoformat()

            quote = {
                "ticker": ticker,
                "price": latest_close,
                "change_pct": change_pct,
                "as_of": as_of,
            }
            return quote, frame[["Date", "Close"]].copy()

        option_columns = {"Last Price", "% Change", "Last Trade Date (EDT)"}
        if option_columns.issubset(columns):
            frame["Last Price"] = pd.to_numeric(frame["Last Price"], errors="coerce")
            frame["pct_change_numeric"] = frame["% Change"].apply(_to_numeric_percent)
            frame["trade_dt"] = pd.to_datetime(frame["Last Trade Date (EDT)"], errors="coerce")
            frame = frame.dropna(subset=["Last Price", "pct_change_numeric", "trade_dt"])

            if frame.empty:
                raise ValueError(f"{ticker} CSV did not have valid option quote rows")

            latest_row = frame.sort_values("trade_dt").iloc[-1]
            latest_price = float(latest_row["Last Price"])
            change_pct = float(latest_row["pct_change_numeric"])

            quote = {
                "ticker": ticker,
                "price": latest_price,
                "change_pct": change_pct,
                "as_of": latest_row["trade_dt"].date().isoformat(),
            }

            normalized = frame.rename(columns={"trade_dt": "Date", "Last Price": "Close"})
            return quote, normalized[["Date", "Close"]].copy()

        raise ValueError(f"{ticker} CSV is missing required quote columns")

    def get_quote(self, ticker: str):
        ticker = ticker.upper()
        if ticker not in self.csv_paths:
            raise ValueError(f"Unsupported ticker: {ticker}")
        if ticker in self.errors:
            raise RuntimeError(self.errors[ticker])
        return self.quotes[ticker]

    def get_history(self, ticker: str, window: int):
        ticker = ticker.upper()
        if ticker not in self.csv_paths:
            raise ValueError(f"Unsupported ticker: {ticker}")
        if ticker in self.errors:
            raise RuntimeError(self.errors[ticker])
        if window <= 0:
            raise ValueError("window must be a positive integer")

        frame = self.frames[ticker].copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
        frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")

        if frame.empty:
            raise ValueError(f"{ticker} history is empty after parsing")

        frame["return_1d"] = frame["Close"].pct_change().fillna(0.0)

        def to_hedge_bucket(abs_return):
            if abs_return >= 0.02:
                return 100
            if abs_return >= 0.015:
                return 75
            if abs_return >= 0.01:
                return 50
            if abs_return >= 0.005:
                return 25
            return 0

        frame["hedge_intensity"] = frame["return_1d"].abs().apply(to_hedge_bucket)
        sample = frame.tail(window)

        points = [
            {
                "date": row["Date"].date().isoformat(),
                "close": float(row["Close"]),
                "return_1d": float(row["return_1d"]),
                "hedge_intensity": int(row["hedge_intensity"]),
            }
            for _, row in sample.iterrows()
        ]
        return {"ticker": ticker, "points": points}


QUOTE_SERVICE = QuoteService(TICKER_CSV_MAP)


def load_paper_orders(path: Path):
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def save_paper_orders(path: Path, orders: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(orders, handle)
    temp_path.replace(path)


PAPER_ORDERS = load_paper_orders(PAPER_ORDERS_PATH)


class HedgeRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "backend_version": "contract-count-v2",
                    "loaded_models": sorted(MODEL_SERVICES.keys()),
                    "model_load_errors": MODEL_LOAD_ERRORS,
                    "feature_counts": {
                        ticker: len(service.all_features)
                        for ticker, service in MODEL_SERVICES.items()
                },
                "routes": [
                    "/api/health",
                    "/api/quote?ticker=SPY",
                    "/api/history?ticker=SPY&window=15",
                    "/api/latest-features?ticker=SPY",
                    "/api/orders/paper?limit=20",
                    "/api/hedge/recommend (POST)",
                    ],
                },
            )
            return

        if parsed.path == "/api/quote":
            query_params = parse_qs(parsed.query)
            ticker = (query_params.get("ticker", [""])[0] or "").upper()
            if not ticker:
                self._send_json(400, {"error": "Missing required query param: ticker"})
                return
            try:
                self._send_json(200, QUOTE_SERVICE.get_quote(ticker))
            except ValueError as err:
                self._send_json(400, {"error": str(err)})
            except RuntimeError as err:
                self._send_json(500, {"error": f"Quote source unavailable: {err}"})
            except Exception as err:
                self._send_json(500, {"error": f"Quote lookup failed: {err}"})
            return

        if parsed.path == "/api/history":
            query_params = parse_qs(parsed.query)
            ticker = (query_params.get("ticker", [""])[0] or "").upper()
            if not ticker:
                self._send_json(400, {"error": "Missing required query param: ticker"})
                return
            raw_window = query_params.get("window", ["15"])[0]
            try:
                window = int(raw_window)
            except ValueError:
                self._send_json(400, {"error": "window must be an integer"})
                return

            try:
                self._send_json(200, QUOTE_SERVICE.get_history(ticker, window))
            except ValueError as err:
                self._send_json(400, {"error": str(err)})
            except RuntimeError as err:
                self._send_json(500, {"error": f"History source unavailable: {err}"})
            except Exception as err:
                self._send_json(500, {"error": f"History lookup failed: {err}"})
            return

        if parsed.path == "/api/latest-features":
            query_params = parse_qs(parsed.query)
            ticker = (query_params.get("ticker", [""])[0] or "").upper()
            if not ticker:
                self._send_json(400, {"error": "Missing required query param: ticker"})
                return
            try:
                self._send_json(200, load_latest_feature_row(ticker))
            except ValueError as err:
                self._send_json(400, {"error": str(err)})
            except Exception as err:
                self._send_json(500, {"error": f"Could not load latest features: {err}"})
            return

        if parsed.path == "/api/orders/paper":
            query_params = parse_qs(parsed.query)
            raw_limit = query_params.get("limit", ["20"])[0]
            try:
                limit = int(raw_limit)
            except ValueError:
                self._send_json(400, {"error": "limit must be an integer"})
                return
            if limit <= 0:
                self._send_json(400, {"error": "limit must be a positive integer"})
                return

            self._send_json(200, {"orders": PAPER_ORDERS[-limit:]})
            return

        self._send_json(404, {"error": "Not found", "path": parsed.path})

    def do_POST(self):
        if self.path == "/api/hedge/recommend":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(raw) if raw else {}

                ticker = str(payload.get("ticker", "SPY")).upper()
                if ticker not in SUPPORTED_TICKERS:
                    raise ValueError(f"Unsupported ticker: {ticker}")
                if ticker not in MODEL_SERVICES:
                    raise RuntimeError(f"No loaded model available for ticker: {ticker}")

                use_latest = bool(payload.get("use_latest_portfolio_row", False))
                contract_count = float(payload.get("contract_count", 1) or 1)
                current_hedge = float(payload.get("current_hedge_shares", 0) or 0)

                if use_latest:
                    latest_row = load_latest_feature_row(ticker)

                    # Recompute live portfolio sizing from current option delta and contract count
                    option_delta = float(latest_row.get("delta", 0.0))
                    latest_row["portfolio_delta"] = option_delta * 100.0 * contract_count
                    if "gamma" in latest_row:
                        latest_row["portfolio_gamma"] = float(latest_row["gamma"]) * 100.0 * contract_count
                    if "theta" in latest_row:
                        latest_row["portfolio_theta"] = float(latest_row["theta"]) * 100.0 * contract_count
                    if "vega" in latest_row:
                        latest_row["portfolio_vega"] = float(latest_row["vega"]) * 100.0 * contract_count

                    latest_row["contract_count"] = contract_count
                    latest_row["current_hedge_shares"] = current_hedge
                    prediction = MODEL_SERVICES[ticker].predict(latest_row)
                else:
                    payload["contract_count"] = contract_count
                    payload["current_hedge_shares"] = current_hedge
                    prediction = MODEL_SERVICES[ticker].predict(payload)

                self._send_json(200, prediction)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON payload"})
            except ValueError as err:
                self._send_json(400, {"error": str(err)})
            except Exception as err:
                self._send_json(500, {"error": f"Inference failed: {err}"})
            return

        if self.path == "/api/orders/paper":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON payload"})
                return

            ticker = str(payload.get("ticker", "")).upper()
            side = str(payload.get("side", "")).lower()

            try:
                quantity = int(payload.get("quantity", 0))
            except (TypeError, ValueError):
                self._send_json(400, {"error": "quantity must be an integer"})
                return

            try:
                hedge_percent = float(payload.get("hedge_percent", 0))
                price = float(payload.get("price", 0))
            except (TypeError, ValueError):
                self._send_json(400, {"error": "hedge_percent and price must be numeric"})
                return

            if ticker not in SUPPORTED_TICKERS:
                self._send_json(400, {"error": f"Unsupported ticker: {ticker}"})
                return
            if side not in {"buy", "sell"}:
                self._send_json(400, {"error": "side must be buy or sell"})
                return
            if quantity <= 0:
                self._send_json(400, {"error": "quantity must be a positive integer"})
                return
            if price <= 0:
                self._send_json(400, {"error": "price must be positive"})
                return

            order = {
                "order_id": f"paper_{uuid4().hex[:12]}",
                "status": "filled",
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "hedge_percent": hedge_percent,
                "filled_price": round(price, 4),
                "filled_at": pd.Timestamp.now("UTC").isoformat(),
            }

            PAPER_ORDERS.append(order)
            save_paper_orders(PAPER_ORDERS_PATH, PAPER_ORDERS)
            self._send_json(200, order)
            return

        self._send_json(404, {"error": "Not found", "path": self.path})


def parse_args():
    parser = argparse.ArgumentParser(description="Run hedge inference API server.")
    parser.add_argument("--host", default=HOST, help=f"Bind host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Bind port (default: {PORT})")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        server = HTTPServer((args.host, args.port), HedgeRequestHandler)
    except OSError as err:
        if getattr(err, "errno", None) == 48:
            server = HTTPServer((args.host, 0), HedgeRequestHandler)
            actual_port = server.server_address[1]
            print(f"Requested port {args.port} is in use; using available port {actual_port} instead.")
        else:
            raise

    actual_port = server.server_address[1]
    print(f"Hedge API server running on http://{args.host}:{actual_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()