"""Send fresh synthetic measurements to the local API; no real measurements are used."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from pregnancy_monitor.data import synthetic_data


def demo_payload(now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc)).floor("5min")
    frame = synthetic_data(days=1, series_count=1)
    frame["timestamp"] = pd.date_range(end=now, periods=len(frame), freq="5min")
    return {"series_id": "synthetic_demo", "data_kind": "synthetic", "measurements": [
        {"timestamp": row.timestamp.isoformat(), "heart_rate_bpm": row.heart_rate_bpm, "quality": "valid"}
        for row in frame.itertuples()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="outputs/api_demo/response.json")
    parser.add_argument("--start-server", action="store_true", help="Start and stop a private local server for this demo")
    parser.add_argument("--backend", choices=["chronos2", "persistence"], default="chronos2",
                        help="Backend for --start-server; an existing server keeps its own configuration")
    args = parser.parse_args()
    token = os.getenv("PM_API_TOKEN")
    if not token and not args.start_server:
        parser.error("Set PM_API_TOKEN in this terminal to the same value as the server")
    if args.start_server and args.url != "http://127.0.0.1:8000":
        parser.error("--start-server uses http://127.0.0.1:8000; choose --url only for an existing server")
    child = None
    try:
        if args.start_server:
            token = secrets.token_urlsafe(32)
            environment = {**os.environ, "PM_API_TOKEN": token, "PM_BACKEND": args.backend}
            child = subprocess.Popen([sys.executable, "-m", "pregnancy_monitor", "serve"], env=environment)
            deadline = time.monotonic() + 120
            while True:
                if child.poll() is not None:
                    raise RuntimeError("The demo server stopped before it was ready")
                try:
                    ready = Request(args.url + "/readyz", headers={"Authorization": f"Bearer {token}"})
                    with urlopen(ready, timeout=2):
                        break
                except HTTPError as exc:
                    raise RuntimeError(f"Server is not ready (HTTP {exc.code}); check weights and configuration") from exc
                except URLError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Server startup timed out")
                    time.sleep(0.25)
        return send_demo(args, token)
    except RuntimeError as exc:
        print(f"Demo failed: {exc}")
        return 1
    finally:
        if child is not None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def send_demo(args, token):
    request = Request(args.url.rstrip("/") + "/v1/forecasts/heart-rate",
                      data=json.dumps(demo_payload()).encode(),
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()}")
        return 1
    except URLError as exc:
        print(f"Connection failed: {exc.reason}")
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Backend: {result['model']['backend']}; {len(result['points'])} points; horizon {result['horizon_minutes']} min")
    print(f"Synthetic response saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
