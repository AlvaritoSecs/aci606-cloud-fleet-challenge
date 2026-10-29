"""Pruebas de la API con una base SQLite temporal. Ejecutar con: pytest -q

Cubren los criterios de aceptación de la Fase 1:
- más de 100 muestras por robot en el historial;
- el historial sobrevive a un "reinicio" (una aplicación nueva sobre la misma base);
- datos inválidos -> 4xx, nunca 5xx, y el servicio sigue sano después.
"""
from __future__ import annotations

import json
import os
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app import db
from app.main import create_app

SAMPLE = {
    "robot_id": "R01", "timestamp": 1780000000.0, "x": 4.21, "y": 7.82, "speed": 0.63,
    "battery": 71.0, "motor_temp": 48.2, "vibration": 0.43, "motor_current": 3.7,
    "obstacle_distance": 0.42,
}
FIELDS = list(SAMPLE)


@pytest.fixture()
def db_url(tmp_path):
    """SQLite temporal por defecto. Con TEST_DATABASE_URL se prueba contra PostgreSQL real.

    ¡Ojo! Con TEST_DATABASE_URL la tabla `telemetry` se BORRA antes de cada prueba:
    usa una base o rama de pruebas, nunca la de producción.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        return f"sqlite:///{tmp_path / 'test.db'}"
    engine = db.make_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS telemetry"))
    engine.dispose()
    return url


@pytest.fixture()
def client(db_url):
    with TestClient(create_app(db_url)) as c:
        yield c


def post_raw(client: TestClient, payload: dict):
    """Envía JSON tal cual, incluidos NaN/Infinity (json.dumps los escribe como literales)."""
    return client.post("/telemetry", content=json.dumps(payload), headers={"Content-Type": "application/json"})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["persistence"] in {"SQLITE", "POSTGRESQL"}


def test_ingest_keeps_starter_response_keys(client):
    r = client.post("/telemetry", json=SAMPLE)
    assert r.status_code == 200
    assert {"accepted", "robot_id", "server_time", "note"} <= r.json().keys()


def test_history_over_100_in_order(client):
    for i in range(150):
        assert client.post("/telemetry", json={**SAMPLE, "timestamp": 1780000000.0 + i}).status_code == 200
    hist = client.get("/robots/R01/history").json()
    assert len(hist) == 150
    assert [h["timestamp"] for h in hist] == sorted(h["timestamp"] for h in hist)
    assert set(FIELDS) <= hist[0].keys()


def test_history_survives_restart(db_url):
    with TestClient(create_app(db_url)) as c:
        for i in range(120):
            c.post("/telemetry", json={**SAMPLE, "robot_id": "R02", "timestamp": 1780000000.0 + i})
        c.post("/telemetry", json={**SAMPLE, "robot_id": "R03"})
    # Proceso nuevo, misma base: equivale a reiniciar el servicio en Render.
    with TestClient(create_app(db_url)) as c:
        assert len(c.get("/robots/R02/history").json()) == 120
        robots = {r["robot_id"]: r for r in c.get("/robots").json()}
        assert set(robots) == {"R02", "R03"}
        assert robots["R02"]["timestamp"] == 1780000000.0 + 119
        assert c.get("/health").json()["robots_in_memory"] == 2


def test_robots_returns_latest_per_robot(client):
    client.post("/telemetry", json={**SAMPLE, "battery": 90.0})
    client.post("/telemetry", json={**SAMPLE, "battery": 80.0})
    client.post("/telemetry", json={**SAMPLE, "robot_id": "R09", "battery": 50.0})
    robots = {r["robot_id"]: r for r in client.get("/robots").json()}
    assert robots["R01"]["battery"] == 80.0 and robots["R09"]["battery"] == 50.0
    assert set(FIELDS) <= robots["R01"].keys()


def test_unknown_robot_is_404(client):
    assert client.get("/robots/NOPE/history").status_code == 404


INVALID = {
    "x NaN": {**SAMPLE, "x": float("nan")},
    "timestamp Infinity": {**SAMPLE, "timestamp": float("inf")},
    "motor_temp NaN": {**SAMPLE, "motor_temp": float("nan")},
    "speed -Infinity": {**SAMPLE, "speed": float("-inf")},
    "motor_temp -500": {**SAMPLE, "motor_temp": -500},
    "motor_temp 1e9": {**SAMPLE, "motor_temp": 1e9},
    "speed 999": {**SAMPLE, "speed": 999},
    "battery 150": {**SAMPLE, "battery": 150},
    "vibration negativa": {**SAMPLE, "vibration": -1},
    "falta vibration": {k: v for k, v in SAMPLE.items() if k != "vibration"},
    "speed texto": {**SAMPLE, "speed": "abc"},
    "robot_id vacío": {**SAMPLE, "robot_id": ""},
    "robot_id de 21": {**SAMPLE, "robot_id": "X" * 21},
    "cuerpo vacío": {},
}


@pytest.mark.parametrize("name", list(INVALID))
def test_invalid_is_4xx_and_service_stays_healthy(client, name):
    r = post_raw(client, INVALID[name])
    assert 400 <= r.status_code < 500, (name, r.status_code, r.text)
    r.json()  # la respuesta de error también es JSON válido
    assert client.get("/robots").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_invalid_json_is_4xx(client):
    r = client.post("/telemetry", content="{no es json", headers={"Content-Type": "application/json"})
    assert 400 <= r.status_code < 500


def test_invalid_samples_are_not_stored(client):
    for payload in INVALID.values():
        post_raw(client, payload)
    assert client.get("/robots").json() == []


def test_dashboard_served(client):
    r = client.get("/dashboard")
    assert r.status_code == 200 and "<html" in r.text.lower()


def test_ingest_is_reasonably_fast(client):
    t0 = time.perf_counter()
    for _ in range(50):
        client.post("/telemetry", json=SAMPLE)
    assert (time.perf_counter() - t0) / 50 < 0.2


def test_history_limit_returns_last_n_in_order(client):
    for i in range(30):
        client.post("/telemetry", json={**SAMPLE, "timestamp": 1780000000.0 + i})
    last = client.get("/robots/R01/history?limit=5").json()
    assert [h["timestamp"] for h in last] == [1780000000.0 + i for i in range(25, 30)]
    assert len(client.get("/robots/R01/history").json()) == 30  # sin limit: historial completo
    assert client.get("/robots/R01/history?limit=0").status_code == 422
