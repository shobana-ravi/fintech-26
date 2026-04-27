import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hedge_xgb.joblib"
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
        if self.path == "/api/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model_path": str(MODEL_PATH),
                    "feature_count": len(MODEL_SERVICE.features),
                },
            )
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/hedge/recommend":
            self._send_json(404, {"error": "Not found"})
            return

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
