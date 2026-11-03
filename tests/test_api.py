import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from threading import Event
import unittest

try:
    from fastapi.testclient import TestClient
    from pregnancy_monitor.api import ApiSettings, MAX_BODY_BYTES, ServiceBoundaryMiddleware, create_app
except ImportError:
    raise unittest.SkipTest("Install the API test dependencies with: python -m pip install -e '.[test]'")

from pregnancy_monitor.forecast import PersistenceForecaster

TOKEN = "synthetic-test-credential-" + "x" * 32
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def payload(origin=NOW):
    return {"series_id": "synthetic_test", "data_kind": "synthetic", "measurements": [
        {"timestamp": (origin - timedelta(minutes=5 * (287 - i))).isoformat(),
         "heart_rate_bpm": 70.0 + i / 100, "quality": "valid"} for i in range(288)]}


class ApiTests(unittest.TestCase):
    def app(self, factory=PersistenceForecaster):
        return create_app(ApiSettings(token=TOKEN, backend="persistence"), factory, lambda: NOW)

    def test_valid_forecast_and_model_loaded_once(self):
        calls = []

        def factory():
            calls.append(1)
            return PersistenceForecaster()

        with TestClient(self.app(factory)) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
            self.assertEqual(client.get("/readyz", headers=HEADERS).status_code, 200)
            for _ in range(2):
                response = client.post("/v1/forecasts/heart-rate", json=payload(), headers=HEADERS)
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(len(result["points"]), 12)
                self.assertEqual(result["horizon_minutes"], 60)
                self.assertEqual(result["model"]["backend"], "persistence")
                self.assertIsNone(result["points"][0]["p10"])
                self.assertTrue(result["points"][-1]["timestamp"].startswith("2026-11-04T13:00:00"))
        self.assertEqual(len(calls), 1)

    def test_unauthorized_precedes_json_parsing(self):
        with TestClient(self.app()) as client:
            response = client.post("/v1/forecasts/heart-rate", content="broken", headers={"Content-Type": "application/json"})
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers["www-authenticate"], "Bearer")
            self.assertEqual(client.get("/readyz").status_code, 401)

    def test_large_allowed_history_uses_only_last_day_and_excess_samples_refused(self):
        histories_seen = []

        class Spy(PersistenceForecaster):
            def predict(self, histories, horizon):
                histories_seen.extend(histories)
                return super().predict(histories, horizon)

        body = payload()
        earlier = payload(NOW - timedelta(days=1))["measurements"]
        for sample in earlier:
            sample["heart_rate_bpm"] = 500.0
        body["measurements"] = earlier + body["measurements"]
        with TestClient(self.app(Spy)) as client:
            response = client.post("/v1/forecasts/heart-rate", json=body, headers=HEADERS)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["input_points"], 576)
            self.assertEqual(response.json()["history_points_used"], 288)
            self.assertEqual(len(histories_seen[0]), 288)
            self.assertLess(max(histories_seen[0]), 100)
            body["measurements"].append(body["measurements"][-1])
            response = client.post("/v1/forecasts/heart-rate", json=body, headers=HEADERS)
            self.assertEqual(response.status_code, 422)
            self.assertEqual(len(histories_seen), 1)

    def test_stale_future_quality_gaps_and_duplicates_refused_before_inference(self):
        calls = []

        class Spy(PersistenceForecaster):
            def predict(self, histories, horizon):
                calls.append(1)
                return super().predict(histories, horizon)

        cases = [(payload(NOW - timedelta(minutes=16)), "stale_observations"),
                 (payload(NOW + timedelta(seconds=61)), "future_observations")]
        for value in ("invalid", "unknown"):
            body = payload()
            body["measurements"][0]["quality"] = value
            cases.append((body, "signal_quality_rejected"))
        for timestamp in (NOW - timedelta(days=2), NOW):
            body = payload()
            body["measurements"][0]["timestamp"] = timestamp.isoformat()
            cases.append((body, "invalid_history"))
        with TestClient(self.app(Spy)) as client:
            for body, code in cases:
                response = client.post("/v1/forecasts/heart-rate", json=body, headers=HEADERS)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["error"]["code"], code)
        self.assertEqual(calls, [])

    def test_schema_errors_do_not_echo_inputs(self):
        cases = []
        body = payload()
        body["unexpected_patient_name"] = "SENSITIVE_SENTINEL"
        cases.append(body)
        for timestamp in ("2026-11-04T12:00:00", "0", 0):
            body = payload()
            body["measurements"][0]["timestamp"] = timestamp
            cases.append(body)
        for value in ("NaN", True, -1):
            body = payload()
            body["measurements"][0]["heart_rate_bpm"] = value
            cases.append(body)
        body = payload()
        body["measurements"] = body["measurements"][:287]
        cases.append(body)
        with TestClient(self.app()) as client:
            for body in cases:
                response = client.post("/v1/forecasts/heart-rate", json=body, headers=HEADERS)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn("SENSITIVE_SENTINEL", response.text)
                self.assertNotIn("measurements", response.text)

    def test_invalid_json_media_type_and_body_limit(self):
        with TestClient(self.app()) as client:
            for content, extra, expected in (("{", {"Content-Type": "application/json"}, 400),
                                             ("{}", {"Content-Type": "text/plain"}, 415),
                                             ("{}", {"Content-Type": "application/json", "Content-Encoding": "gzip"}, 415),
                                             ("x" * (MAX_BODY_BYTES + 1), {"Content-Type": "application/json"}, 413)):
                response = client.post("/v1/forecasts/heart-rate", content=content, headers={**HEADERS, **extra})
                self.assertEqual(response.status_code, expected, response.text)

    def test_stream_limit_without_content_length(self):
        async def check():
            async def downstream(scope, receive, send):
                self.fail("Oversized body reached the application")

            sent = []
            chunks = iter([{"type": "http.request", "body": b"x" * (MAX_BODY_BYTES // 2), "more_body": True},
                           {"type": "http.request", "body": b"x" * (MAX_BODY_BYTES // 2 + 1), "more_body": False}])

            async def receive():
                return next(chunks)

            async def send(message):
                sent.append(message)

            await ServiceBoundaryMiddleware(downstream, TOKEN)(
                {"type": "http", "method": "POST", "path": "/v1/forecasts/heart-rate",
                 "headers": [(b"authorization", f"Bearer {TOKEN}".encode()), (b"content-type", b"application/json")]},
                receive, send)
            self.assertEqual(sent[0]["status"], 413)

        asyncio.run(check())

    def test_model_failure_has_no_silent_fallback_or_exception_leak(self):
        class Broken(PersistenceForecaster):
            def predict(self, histories, horizon):
                raise RuntimeError("SENSITIVE_MODEL_PATH_OR_DATA")

        with TestClient(self.app(Broken)) as client:
            response = client.post("/v1/forecasts/heart-rate", json=payload(), headers=HEADERS)
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("SENSITIVE_MODEL_PATH_OR_DATA", response.text)
            self.assertEqual(client.get("/readyz", headers=HEADERS).status_code, 503)

        def missing():
            raise OSError("SENSITIVE_MODEL_PATH")

        with TestClient(self.app(missing)) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
            self.assertEqual(client.get("/readyz", headers=HEADERS).status_code, 503)

    def test_concurrent_inference_is_rejected_without_waiting(self):
        entered, release = Event(), Event()

        class Blocking(PersistenceForecaster):
            def predict(self, histories, horizon):
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("Test did not release inference")
                return super().predict(histories, horizon)

        with TestClient(self.app(Blocking)) as client, ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(client.post, "/v1/forecasts/heart-rate", json=payload(), headers=HEADERS)
            try:
                self.assertTrue(entered.wait(timeout=3))
                response = client.post("/v1/forecasts/heart-rate", json=payload(), headers=HEADERS)
                self.assertEqual(response.status_code, 429, response.text)
                self.assertEqual(response.headers["retry-after"], "1")
            finally:
                release.set()
            self.assertEqual(first.result(timeout=3).status_code, 200)

    def test_openapi_and_token_configuration(self):
        schema = self.app().openapi()
        self.assertIn("/v1/forecasts/heart-rate", schema["paths"])
        self.assertIn("HTTPBearer", schema["components"]["securitySchemes"])
        with self.assertRaises(ValueError):
            ApiSettings(token="short")


if __name__ == "__main__":
    unittest.main()
