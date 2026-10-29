from __future__ import annotations

import logging
import math
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app import db
from app.metrics import Metrics
from app.model import ModelService
from app.schemas import PredictionRequest, Telemetry

log = logging.getLogger("aci606")

VERSION = "0.4.0"
DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"

router = APIRouter()


def _persistence_name(engine) -> str:
    return {"postgresql": "POSTGRESQL", "sqlite": "SQLITE"}.get(engine.dialect.name, engine.dialect.name.upper())


@router.get("/health")
def health(request: Request) -> dict:
    # No consulta la base en cada llamada: Render llama /health seguido y cada consulta
    # mantendría a Neon despierta, gastando la cuota mensual de cómputo.
    state = request.app.state
    return {
        "status": "ok",
        "service": "aci606-cloud-fleet",
        "version": VERSION,
        "robots_in_memory": len(state.cache),
        "utc": datetime.now(timezone.utc).isoformat(),
        "persistence": _persistence_name(state.engine),
        "database": state.db_status,
        "model_loaded": state.model.ready,
        "model_version": state.model.version if state.model.ready else None,
    }


@router.post("/telemetry")
def ingest_telemetry(sample: Telemetry, request: Request) -> dict:
    state = request.app.state
    data = sample.model_dump()
    # La predicción se guarda junto con la muestra: el panel y el historial la muestran
    # sin volver a llamar al modelo. Si no hay modelo, la muestra se guarda igual.
    prediction = None
    if state.model.ready:
        p = state.model.predict(data)
        prediction = {"predicted_status": p["predicted_status"], "confidence": p["confidence"],
                      "model_version": state.model.version}
    row = db.insert_sample(state.engine, data, prediction)
    state.cache.update(row)
    state.db_status = "ok"
    state.telemetry_count += 1
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "server_time": row["received_at"],
        "note": f"Stored in {_persistence_name(state.engine)}.",
        "predicted_status": row.get("predicted_status"),
    }


@router.get("/robots")
def list_robots(request: Request) -> list[dict]:
    state = request.app.state
    state.cache.ensure_loaded(state.engine)
    return state.cache.all()


@router.get("/robots/{robot_id}/history")
def robot_history(robot_id: str, request: Request) -> list[dict]:
    rows = db.fetch_history(request.app.state.engine, robot_id)
    request.app.state.db_status = "ok"
    if not rows:
        raise HTTPException(status_code=404, detail="Robot not found")
    return rows


@router.post("/predict")
def predict(sample: PredictionRequest, request: Request) -> dict:
    # Solo usa el modelo en memoria: no toca la base, así su latencia no depende de Neon.
    state = request.app.state
    if not state.model.ready:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_PATH and the deploy logs.")
    p = state.model.predict(sample.model_dump())
    state.metrics.record_prediction(p["predicted_status"], p["inference_ms"])
    return {
        "robot_id": sample.robot_id,
        "predicted_status": p["predicted_status"],
        "confidence": p["confidence"],
        "model_version": state.model.version,
        "inference_ms": p["inference_ms"],
    }


@router.get("/metrics")
def metrics(request: Request) -> dict:
    state = request.app.state
    return {
        # Claves del starter (se mantienen).
        "requests_received": state.telemetry_count,
        "robots_in_memory": len(state.cache),
        "persistence": _persistence_name(state.engine),
        "prediction_service": "READY" if state.model.ready else "MODEL_NOT_LOADED",
        # Estado de dependencias.
        "database": state.db_status,
        "model": state.model.info(),
        # Contadores y latencias desde el último arranque.
        **state.metrics.snapshot(),
        "note": "Counters reset when the service restarts; telemetry is persisted in the database.",
    }


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD.read_text(encoding="utf-8")


def _clean(value):
    """Hace serializable un valor que viene del cliente (p. ej. NaN dentro de un error 422)."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    # El manejador por defecto de FastAPI devuelve el valor recibido; si era NaN o Infinity,
    # la respuesta no se puede serializar como JSON y el 422 se convierte en un 500.
    errors = [
        {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", ""),
         "input": _clean(e.get("input"))}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    request.app.state.db_status = "error"
    log.exception("Database error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=503, content={"detail": "Database temporarily unavailable. Retry shortly."})


def create_app(database_url: str | None = None, model_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL no está definida. En Render: Environment -> DATABASE_URL. "
                "En local: export DATABASE_URL=sqlite:///./local.db"
            )
        app.state.model.load()
        app.state.engine = db.make_engine(url)
        db.init_db(app.state.engine)
        try:
            app.state.cache.ensure_loaded(app.state.engine)
            app.state.db_status = "ok"
        except SQLAlchemyError:
            log.exception("Could not preload latest samples; will retry on first /robots")
            app.state.db_status = "error"
        yield
        app.state.engine.dispose()

    app = FastAPI(
        title="ACI606 Cloud Fleet Challenge",
        version=VERSION,
        description="Telemetría persistente, inferencia y métricas para una flota virtual de robots.",
        lifespan=lifespan,
    )
    app.state.cache = db.LatestCache()
    app.state.model = ModelService(model_path)
    app.state.db_status = "unknown"
    app.state.telemetry_count = 0
    app.state.metrics = Metrics()

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        t0 = time.perf_counter()
        status = 500  # si la solicitud termina en una excepción no controlada
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            name = f"{request.method} {route.path}" if route is not None else f"{request.method} (sin ruta)"
            app.state.metrics.record_request(name, status, (time.perf_counter() - t0) * 1000)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(SQLAlchemyError, database_error_handler)
    app.include_router(router)
    return app


app = create_app()
