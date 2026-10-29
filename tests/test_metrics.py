"""Pruebas de GET /metrics. Ejecutar con: pytest -q"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

SAMPLE = {
    "robot_id": "R01", "timestamp": 1780000000.0, "x": 4.21, "y": 7.82, "speed": 0.63,
    "battery": 71.0, "motor_temp": 48.2, "vibration": 0.43, "motor_current": 3.7,
    "obstacle_distance": 0.42,
}
# Exigidas por el README del starter (sección Observability) y por el PDF.
REQUIRED = {"requests_received", "telemetry_requests", "prediction_requests", "errors_4xx",
            "errors_5xx", "avg_prediction_latency_ms", "uptime_seconds", "predict_latency_ms"}
STARTER_KEYS = {"requests_received", "robots_in_memory", "persistence", "prediction_service"}


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(f"sqlite:///{tmp_path / 'm.db'}")) as c:
        yield c


def test_metrics_has_required_and_starter_keys(client):
    body = client.get("/metrics").json()
    assert REQUIRED <= body.keys()
    assert STARTER_KEYS <= body.keys()
    assert body["prediction_service"] == "READY"


def test_counters_follow_requests(client):
    for _ in range(3):
        client.post("/telemetry", json=SAMPLE)
    for _ in range(5):
        client.post("/predict", json=SAMPLE)
    client.post("/predict", content=json.dumps({**SAMPLE, "x": float("nan")}),
                headers={"Content-Type": "application/json"})
    client.get("/robots/NOPE/history")
    m = client.get("/metrics").json()
    assert m["requests_received"] == 3
    assert m["telemetry_requests"] == 3
    assert m["prediction_requests"] == 6
    assert m["prediction_rejected_4xx"] == 1
    assert m["errors_4xx"] == 2          # el 422 y el 404
    assert m["errors_5xx"] == 0
    assert m["predict_latency_ms"]["count"] == 5
    assert m["inference_ms"]["count"] == 5
    assert m["avg_prediction_latency_ms"] > 0
    assert m["predict_latency_ms"]["p95"] >= m["predict_latency_ms"]["p50"]
    assert sum(m["predicted_classes"].values()) == 5
    assert m["requests_by_route"]["GET /robots/{robot_id}/history"]["404"] == 1


def test_uptime_grows(client):
    a = client.get("/metrics").json()["uptime_seconds"]
    b = client.get("/metrics").json()["uptime_seconds"]
    assert b >= a >= 0


def test_metrics_is_valid_json_without_predictions(client):
    m = client.get("/metrics").json()
    assert m["avg_prediction_latency_ms"] is None
    assert m["predict_latency_ms"]["count"] == 0
