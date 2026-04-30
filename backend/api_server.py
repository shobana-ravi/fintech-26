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

MODEL_PATH = _REPO_ROOT / "outputs" / "xgboost_hedge_bundle.joblib"
SPY_TRAINING_STATS_CSV = _REPO_ROOT / "data" / "SPY" / "spy_training_dataset.csv"
DIA_CSV_PATH = _REPO_ROOT / "data" / "dia_us_d.csv"
IWM_CSV_PATH = _REPO_ROOT / "data" / "IWM_data.csv"
QQQ_CSV_PATH = _REPO_ROOT / "data" / "QQQ" / "qqq_us_d.csv"
SPY_CSV_PATH = _REPO_ROOT / "data" / "SPY" / "spy_us_d.csv"
PAPER_ORDERS_PATH = _REPO_ROOT / "data" / "paper_orders.json"
HOST = "0.0.0.0"
PORT = 8001
DEBUG_LOG_PATH = _REPO_ROOT / "outputs" / "agent_debug.log"


def debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    #region agent log
    payload = {
        "sessionId": "d5969c",
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
    #endregion


class ModelService:
    def __init__(self, model_path: Path):
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

        user_portfolio_delta = float(payload["portfolio_delta"])

        row = {c: float(payload[c]) for c in self.feature_cols}
        # Training rows use BS option Greeks and portfolio_* = greek * 100; placeholders
        # from the client would skew predictions (often always 0% hedge).
        apply_training_style_option_row(row)

        if abs(float(row["option_pnl"])) < 1e-8:
            row["option_pnl"] = SPY_TRAINING_MEDIAN_OPTION_PNL

        self._clip_returns_like_training(row)

        frame = pd.DataFrame([row])
        frame = engineer_features(frame)
        scored = frame[self.all_features].fillna(0)

        predicted_enc = int(self.model.predict(scored)[0])
        probabilities = np.asarray(self.model.predict_proba(scored)[0], dtype=float)
        confidence = float(probabilities.max())
        predicted_bucket = float(
            encoded_prediction_to_bucket(self.label_encoder, predicted_enc)
        )

        # Live rows can sit outside the training joint distribution; the classifier
        # may assign extreme mass to the 0% bucket. When returns or vol look active,
        # nudge toward a probability blend so sizing is not stuck at zero every refresh.
        active_market = abs(float(row["return_1d"])) > 0.008 or float(
            row["realized_vol_20d"]
        ) > 0.12
        if predicted_bucket == 0.0 and confidence >= 0.92 and active_market:
            alpha = 0.35
            n_classes = len(probabilities)
            blend = (1.0 - alpha) * probabilities + alpha * (
                np.ones(n_classes, dtype=float) / n_classes
            )
            soft = sum(
                blend[i]
                * float(encoded_prediction_to_bucket(self.label_encoder, i))
                for i in range(n_classes)
            )
            lattice = [0.0, 0.25, 0.5, 0.75, 1.0]
            adjusted = float(min(lattice, key=lambda b: abs(b - soft)))
            if adjusted > 0.0:
                predicted_bucket = adjusted
                for enc_idx in range(n_classes):
                    if (
                        abs(
                            encoded_prediction_to_bucket(self.label_encoder, enc_idx)
                            - predicted_bucket
                        )
                        < 1e-6
                    ):
                        predicted_enc = enc_idx
                        break

        current_hedge = float(payload.get("current_hedge_shares", 0) or 0)
        target_hedge_shares = -user_portfolio_delta * predicted_bucket
        shares_to_trade = target_hedge_shares - current_hedge
        action = format_trade_action(shares_to_trade)

        return {
            "predicted_hedge_class": predicted_enc,
            "predicted_hedge_ratio_bucket": predicted_bucket,
            "prediction_confidence": confidence,
            "features_used": self.all_features,
            "base_features": self.feature_cols,
            "ticker": payload.get("ticker", "UNKNOWN"),
            "portfolio_delta": user_portfolio_delta,
            "current_hedge_shares": current_hedge,
            "target_hedge_shares": target_hedge_shares,
            "shares_to_trade": shares_to_trade,
            "action": action,
        }


MODEL_SERVICE = ModelService(MODEL_PATH)


def _to_numeric_percent(value):
    if pd.isna(value):
        return None
    cleaned = str(value).replace("%", "").strip()
    if cleaned in {"", "-"}:
        return None
    return float(cleaned)


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
        # Some exports include a prefixed "git aDate" header; normalize it.
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
            # Keep a consistent Date/Close shape for downstream history logic.
            normalized = frame.rename(columns={"trade_dt": "Date", "Last Price": "Close"})
            return quote, normalized[["Date", "Close"]].copy()

        raise ValueError(f"{ticker} CSV is missing required quote columns")

    def get_quote(self, ticker: str):
        normalized = ticker.upper()
        if normalized not in self.csv_paths:
            raise ValueError(f"Unsupported ticker: {normalized}")
        if normalized in self.errors:
            raise RuntimeError(self.errors[normalized])
        return self.quotes[normalized]

    def get_history(self, ticker: str, window: int):
        normalized = ticker.upper()
        if normalized not in self.csv_paths:
            raise ValueError(f"Unsupported ticker: {normalized}")
        if normalized in self.errors:
            raise RuntimeError(self.errors[normalized])
        if window <= 0:
            raise ValueError("window must be a positive integer")

        frame = self.frames[normalized].copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
        frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
        if frame.empty:
            raise ValueError(f"{normalized} history is empty after parsing")

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
        return {"ticker": normalized, "points": points}


QUOTE_SERVICE = QuoteService(
    {
        "DIA": DIA_CSV_PATH,
        "IWM": IWM_CSV_PATH,
        "QQQ": QQQ_CSV_PATH,
        "SPY": SPY_CSV_PATH,
    }
)


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
SUPPORTED_TICKERS = {"SPY", "QQQ", "DIA", "IWM"}


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
                    "model_path": str(MODEL_PATH),
                    "feature_count": len(MODEL_SERVICE.all_features),
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
                quote = QUOTE_SERVICE.get_quote(ticker)
                self._send_json(200, quote)
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
                history = QUOTE_SERVICE.get_history(ticker, window)
                self._send_json(200, history)
            except ValueError as err:
                self._send_json(400, {"error": str(err)})
            except RuntimeError as err:
                self._send_json(500, {"error": f"History source unavailable: {err}"})
            except Exception as err:
                self._send_json(500, {"error": f"History lookup failed: {err}"})
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

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/api/hedge/recommend":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(raw) if raw else {}
                prediction = MODEL_SERVICE.predict(payload)
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

        self._send_json(404, {"error": "Not found"})


def parse_args():
    parser = argparse.ArgumentParser(description="Run hedge inference API server.")
    parser.add_argument("--host", default=HOST, help=f"Bind host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Bind port (default: {PORT})")
    return parser.parse_args()


def main():
    args = parse_args()
    #region agent log
    debug_log(
        run_id="pre-fix",
        hypothesis_id="H3",
        location="backend/api_server.py:main",
        message="Parsed startup arguments",
        data={"pid": os.getpid(), "host": args.host, "port": args.port},
    )
    #endregion
    try:
        server = HTTPServer((args.host, args.port), HedgeRequestHandler)
    except OSError as err:
        #region agent log
        debug_log(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="backend/api_server.py:main",
            message="Socket bind failed",
            data={
                "pid": os.getpid(),
                "host": args.host,
                "port": args.port,
                "errno": getattr(err, "errno", None),
                "error": str(err),
            },
        )
        #endregion
        if getattr(err, "errno", None) == 48:
            #region agent log
            debug_log(
                run_id="pre-fix",
                hypothesis_id="H5",
                location="backend/api_server.py:main",
                message="Retrying bind on OS-assigned free port",
                data={"pid": os.getpid(), "host": args.host, "requested_port": args.port},
            )
            #endregion
            server = HTTPServer((args.host, 0), HedgeRequestHandler)
            actual_port = server.server_address[1]
            #region agent log
            debug_log(
                run_id="pre-fix",
                hypothesis_id="H5",
                location="backend/api_server.py:main",
                message="Fallback bind succeeded",
                data={"pid": os.getpid(), "host": args.host, "actual_port": actual_port},
            )
            #endregion
            print(
                f"Requested port {args.port} is in use; using available port {actual_port} instead."
            )
        else:
            raise

    #region agent log
    debug_log(
        run_id="pre-fix",
        hypothesis_id="H4",
        location="backend/api_server.py:main",
        message="Server bind succeeded",
        data={"pid": os.getpid(), "host": args.host, "port": args.port},
    )
    #endregion
    print(f"Hedge API server running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
