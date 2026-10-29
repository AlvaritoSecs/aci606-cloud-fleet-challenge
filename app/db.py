"""Persistencia de la telemetría en PostgreSQL (Neon) mediante SQLAlchemy Core.

La base de datos es la fuente de verdad: cada muestra válida se guarda como una fila
de la tabla `telemetry`, así que el historial sobrevive a reinicios y redeploys de Render
(cuyo disco es efímero).

La URL llega por la variable de entorno DATABASE_URL; nunca se escribe en el código.
Para desarrollo local sin Neon se puede usar DATABASE_URL=sqlite:///./local.db
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Column, DateTime, Double, Engine, Index, Integer, MetaData, String, Table,
    create_engine, func, insert, select,
)

metadata = MetaData()

telemetry = Table(
    "telemetry", metadata,
    # BIGINT autoincremental en PostgreSQL; INTEGER en SQLite (requisito de su autoincremento).
    Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
    Column("robot_id", String(20), nullable=False),
    Column("timestamp", Double, nullable=False),
    Column("x", Double, nullable=False),
    Column("y", Double, nullable=False),
    Column("speed", Double, nullable=False),
    Column("battery", Double, nullable=False),
    Column("motor_temp", Double, nullable=False),
    Column("vibration", Double, nullable=False),
    Column("motor_current", Double, nullable=False),
    Column("obstacle_distance", Double, nullable=False),
    # Se llenan desde la Fase 3, cuando /telemetry guarde también la predicción.
    Column("predicted_status", String(10), nullable=True),
    Column("confidence", Double, nullable=True),
    Column("model_version", String(32), nullable=True),
    Column("received_at", DateTime(timezone=True), nullable=False),
    # Permite leer el historial de un robot en orden sin recorrer toda la tabla.
    Index("ix_telemetry_robot_id_id", "robot_id", "id"),
)

PUBLIC_COLUMNS = [c for c in telemetry.c if c.name != "id"]


def normalize_url(url: str) -> str:
    """Neon y Render entregan `postgresql://` o `postgres://`; SQLAlchemy necesita el driver psycopg 3."""
    url = url.strip()
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def make_engine(url: str) -> Engine:
    url = normalize_url(url)
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(
        url,
        # Neon suspende la base tras 5 min sin uso y corta las conexiones abiertas:
        # pre_ping prueba cada conexión antes de usarla y abre una nueva si está muerta.
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=10,
        pool_timeout=10,
        # Compatible con el endpoint con pooler de Neon (PgBouncer): sin sentencias preparadas.
        connect_args={"prepare_threshold": None, "connect_timeout": 10},
    )


def init_db(engine: Engine, attempts: int = 3) -> None:
    """Crea la tabla si no existe. Reintenta porque Neon puede tardar en despertar."""
    for i in range(attempts):
        try:
            metadata.create_all(engine)
            return
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 * (i + 1))


def _row(mapping) -> dict:
    row = dict(mapping)
    received = row.get("received_at")
    if isinstance(received, datetime):
        if received.tzinfo is None:  # SQLite no guarda zona horaria
            received = received.replace(tzinfo=timezone.utc)
        row["received_at"] = received.isoformat()
    return row


def insert_sample(engine: Engine, sample: dict, prediction: dict | None = None) -> dict:
    # Las columnas de predicción siempre están presentes (None si no hay modelo), así la
    # fila en caché tiene la misma forma que la leída desde la base.
    empty = {"predicted_status": None, "confidence": None, "model_version": None}
    values = {**sample, **empty, **(prediction or {}), "received_at": datetime.now(timezone.utc)}
    with engine.begin() as conn:
        new_id = conn.execute(insert(telemetry).values(**values).returning(telemetry.c.id)).scalar_one()
    return _row({**values, "id": new_id})


def fetch_latest(engine: Engine) -> list[dict]:
    """Última muestra recibida de cada robot (la de mayor id)."""
    last = select(func.max(telemetry.c.id).label("id")).group_by(telemetry.c.robot_id).subquery()
    stmt = select(telemetry).join(last, telemetry.c.id == last.c.id).order_by(telemetry.c.robot_id)
    with engine.connect() as conn:
        return [_row(r) for r in conn.execute(stmt).mappings()]


def fetch_history(engine: Engine, robot_id: str) -> list[dict]:
    """Historial completo de un robot, en orden de llegada. Sin límite: el PDF exige el historial completo."""
    stmt = select(*PUBLIC_COLUMNS).where(telemetry.c.robot_id == robot_id).order_by(telemetry.c.id)
    with engine.connect() as conn:
        return [_row(r) for r in conn.execute(stmt).mappings()]


class LatestCache:
    """Última muestra de cada robot, en memoria.

    Evita consultar la base cada vez que el panel pide /robots (cada pocos segundos).
    No es una segunda fuente de verdad: se reconstruye desde la base al arrancar y solo se
    actualiza DESPUÉS de que una muestra quedó guardada en la base.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self, engine: Engine) -> None:
        if self._loaded:
            return
        rows = fetch_latest(engine)
        with self._lock:
            for row in rows:
                current = self._data.get(row["robot_id"])
                if current is None or row["id"] > current["id"]:
                    self._data[row["robot_id"]] = row
            self._loaded = True

    def update(self, row: dict) -> None:
        with self._lock:
            current = self._data.get(row["robot_id"])
            if current is None or row["id"] > current["id"]:
                self._data[row["robot_id"]] = row

    def all(self) -> list[dict]:
        with self._lock:
            rows = sorted(self._data.values(), key=lambda r: r["robot_id"])
        return [{k: v for k, v in r.items() if k != "id"} for r in rows]

    def __len__(self) -> int:
        return len(self._data)
