import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hedge_xgb.joblib"
DIA_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "dia_us_d.csv"
IWM_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "IWM_data.csv"
QQQ_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "qqq_us_d.csv"
SPY_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "spy_us_d.csv"
PAPER_ORDERS_PATH = Path(__file__).resolve().parents[1] / "data" / "paper_orders.json"
HOST = "0.0.0.0"
PORT = 8000
DEBUG_LOG_PATH = Path("/Users/boppa/fintech-26/.cursor/debug-d5969c.log")


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
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    #endregion


class ModelService:
    def __init__(self, model_path: Path):
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.features = self.bundle["features"]
        self.class_to_bucket = {
            int(k): float(v) for k, v in self.bundle["class_to_bucket"].items()
        }

    def predict(self, payload: dict) -> dict:
        missing = [feature for feature in self.features if feature not in payload]
        if missing:
            raise ValueError(f"Missing required feature fields: {', '.join(missing)}")

        row = {feature: float(payload[feature]) for feature in self.features}
        frame = pd.DataFrame([row], columns=self.features)

        predicted_class = int(self.model.predict(frame)[0])
        probabilities = self.model.predict_proba(frame)[0]
        confidence = float(probabilities.max())
        predicted_bucket = self.class_to_bucket[predicted_class]

        return {
            "predicted_hedge_class": predicted_class,
            "predicted_hedge_ratio_bucket": predicted_bucket,
            "prediction_confidence": confidence,
            "features_used": self.features,
            "ticker": payload.get("ticker", "UNKNOWN"),
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
                    "feature_count": len(MODEL_SERVICE.features),
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
