"""Private, stateless prototype API. The service credential belongs on a backend."""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
import re
import secrets
from threading import Lock
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException

from .data import ExperimentConfig
from .forecast import MODEL_ID, MODEL_REVISION, ChronosForecaster, PersistenceForecaster, forecast_frame
from .quality import DataQualityError, QualityPolicy, live_history

logger = logging.getLogger(__name__)
MAX_SAMPLES = 576
MAX_BODY_BYTES = 256 * 1024


@dataclass(frozen=True)
class ApiSettings:
    token: str
    backend: str = "chronos2"
    model: str = MODEL_ID
    revision: str = MODEL_REVISION
    device: str = "cpu"
    offline: bool = True
    policy: QualityPolicy = field(default_factory=QualityPolicy)

    def __post_init__(self):
        if len(self.token) < 32 or not self.token.isascii() or any(x.isspace() for x in self.token):
            raise ValueError("PM_API_TOKEN must be a random ASCII token of at least 32 characters without spaces")
        if self.backend not in ("chronos2", "persistence") or self.device not in ("cpu", "cuda"):
            raise ValueError("Invalid PM_BACKEND or PM_DEVICE")

    @classmethod
    def from_env(cls):
        offline = os.getenv("PM_OFFLINE", "1")
        if offline not in ("0", "1"):
            raise ValueError("PM_OFFLINE must be 0 or 1")
        return cls(token=os.getenv("PM_API_TOKEN", ""), backend=os.getenv("PM_BACKEND", "chronos2"),
                   model=os.getenv("PM_MODEL_PATH", MODEL_ID), revision=os.getenv("PM_MODEL_REVISION", MODEL_REVISION),
                   device=os.getenv("PM_DEVICE", "cpu"), offline=offline == "1",
                   policy=QualityPolicy(int(os.getenv("PM_MAX_AGE_MINUTES", "15")),
                                        int(os.getenv("PM_FUTURE_SKEW_SECONDS", "60"))))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Measurement(StrictModel):
    timestamp: AwareDatetime
    heart_rate_bpm: Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
    quality: Literal["valid", "invalid", "unknown"]

    @field_validator("timestamp", mode="before")
    @classmethod
    def iso_timestamp(cls, value):
        if not isinstance(value, str) or len(value) > 64 or not re.search(r"[Tt].*(?:[Zz]|[+-]\d{2}:\d{2})$", value):
            raise ValueError("Use an ISO 8601 string with an explicit timezone")
        return value


class ForecastRequest(StrictModel):
    series_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]
    data_kind: Literal["synthetic", "device"]
    measurements: Annotated[list[Measurement], Field(min_length=288, max_length=MAX_SAMPLES)]


class ForecastPoint(StrictModel):
    timestamp: AwareDatetime
    p10: float | None
    p50: float
    p90: float | None


class ModelInfo(StrictModel):
    backend: str
    revision: str | None
    clinical_validity: Literal["not_evaluated"] = "not_evaluated"


class ForecastResponse(StrictModel):
    series_id: str
    data_kind: Literal["synthetic", "device"]
    generated_at: AwareDatetime
    forecast_origin: AwareDatetime
    interval_minutes: int
    horizon_minutes: int
    input_points: int
    history_points_used: int
    model: ModelInfo
    points: Annotated[list[ForecastPoint], Field(min_length=12, max_length=12)]


class ErrorDetail(StrictModel):
    code: str
    message: str


class ErrorResponse(StrictModel):
    error: ErrorDetail


def error_response(status, code, message, headers=None):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}}, headers=headers)


class ServiceBoundaryMiddleware:
    """Authenticate before reading data and limit streamed bodies, even without Content-Length."""

    def __init__(self, app, token):
        self.app = app
        self.token = token.encode("ascii")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        if scope["path"] not in ("/healthz", "/docs", "/openapi.json", "/docs/oauth2-redirect"):
            scheme, _, credential = headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not secrets.compare_digest(credential.encode(), self.token):
                return await error_response(401, "unauthorized", "A valid service credential is required",
                                            {"WWW-Authenticate": "Bearer"})(scope, receive, send)
        if scope["method"] == "POST" and scope["path"] == "/v1/forecasts/heart-rate":
            if headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                return await error_response(415, "unsupported_media_type", "Use application/json")(scope, receive, send)
            if headers.get("content-encoding", "identity").lower() != "identity":
                return await error_response(415, "unsupported_encoding", "Compressed request bodies are not supported")(scope, receive, send)
            length = headers.get("content-length")
            try:
                if length is not None and int(length) < 0:
                    raise ValueError
                too_large = length is not None and int(length) > MAX_BODY_BYTES
            except ValueError:
                return await error_response(400, "invalid_content_length", "Invalid Content-Length")(scope, receive, send)
            if too_large:
                return await error_response(413, "request_too_large", "Request body exceeds the service limit")(scope, receive, send)
            chunks, size = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > MAX_BODY_BYTES:
                    return await error_response(413, "request_too_large", "Request body exceeds the service limit")(scope, receive, send)
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            received = False

            async def replay():
                nonlocal received
                if not received:
                    received = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            return await self.app(scope, replay, send)
        return await self.app(scope, receive, send)


def create_app(settings=None, forecaster_factory=None, clock=None):
    settings = settings or ApiSettings.from_env()
    config = ExperimentConfig()
    clock = clock or (lambda: datetime.now(timezone.utc))
    inference_lock = Lock()

    def default_factory():
        if settings.backend == "persistence":
            return PersistenceForecaster()
        return ChronosForecaster(settings.model, settings.revision, settings.device, settings.offline)

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = None
        app.state.failed = False
        try:
            app.state.model = await run_in_threadpool(forecaster_factory or default_factory)
        except Exception as exc:
            app.state.failed = True
            logger.warning("Forecast model initialization failed (%s)", type(exc).__name__)
        yield
        app.state.model = None

    app = FastAPI(title="Сохраняя главное — экспериментальный прогноз ЧСС", version="0.2.0",
                  lifespan=lifespan, redoc_url=None)
    app.add_middleware(ServiceBoundaryMiddleware, token=settings.token)
    bearer = HTTPBearer()

    def ready():
        return getattr(app.state, "model", None) is not None and not getattr(app.state, "failed", True)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # FastAPI's default error details can echo entire submitted measurements.
        status = 400 if any(item["type"] == "json_invalid" for item in exc.errors()) else 422
        return error_response(status, "invalid_request", "Request does not match the documented schema")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": "Request failed"}
        return error_response(exc.status_code, detail["code"], detail["message"], exc.headers)

    @app.get("/healthz")
    def health():
        return {"status": "alive"}

    @app.get("/readyz", dependencies=[Depends(bearer)], responses={503: {"model": ErrorResponse}})
    def readiness():
        if not ready():
            return error_response(503, "model_unavailable", "Model is not ready; check weights and dependencies")
        return {"status": "ready", "backend": app.state.model.name}

    @app.post("/v1/forecasts/heart-rate", response_model=ForecastResponse, dependencies=[Depends(bearer)],
              responses={status: {"model": ErrorResponse, "description": description} for status, description in (
                  (400, "Invalid JSON"), (401, "Invalid service credential"), (413, "Body too large"),
                  (415, "Unsupported media type or compression"), (422, "Invalid schema or data quality"),
                  (429, "Inference capacity is busy"), (503, "Model unavailable"))})
    def forecast(payload: ForecastRequest):
        frame = pd.DataFrame({
            "series_id": payload.series_id,
            "timestamp": [x.timestamp.isoformat() for x in payload.measurements],
            "heart_rate_bpm": [x.heart_rate_bpm for x in payload.measurements],
        })
        try:
            validated = live_history(frame, [x.quality for x in payload.measurements], config, settings.policy, clock())
        except DataQualityError as exc:
            raise HTTPException(422, {"code": exc.code, "message": str(exc)}) from exc
        if not ready():
            raise HTTPException(503, {"code": "model_unavailable", "message": "Forecast model is not ready"})
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(429, {"code": "inference_busy", "message": "Retry when the current forecast completes"},
                                headers={"Retry-After": "1"})
        if not ready():
            inference_lock.release()
            raise HTTPException(503, {"code": "model_unavailable", "message": "Forecast model is not ready"})
        try:
            rows = forecast_frame(validated, config, app.state.model)
            points = [ForecastPoint(timestamp=row.timestamp, p10=None if np.isnan(row.p10) else row.p10,
                                    p50=row.p50, p90=None if np.isnan(row.p90) else row.p90)
                      for row in rows.itertuples()]
            metadata = app.state.model.metadata()
        except Exception as exc:
            app.state.failed = True
            logger.warning("Forecast inference failed (%s)", type(exc).__name__)
            raise HTTPException(503, {"code": "inference_failed", "message": "Forecast failed; service restart is required"}) from exc
        finally:
            inference_lock.release()
        return ForecastResponse(
            series_id=payload.series_id, data_kind=payload.data_kind, generated_at=clock(),
            forecast_origin=validated.timestamp.iloc[-1], interval_minutes=config.interval_minutes,
            horizon_minutes=config.interval_minutes * config.horizon_steps, input_points=len(validated),
            history_points_used=config.history_steps,
            model=ModelInfo(backend=metadata["backend"], revision=metadata.get("revision")), points=points,
        )

    return app
