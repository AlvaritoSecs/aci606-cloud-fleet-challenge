"""Pruebas de POST /predict y de la predicción guardada con la telemetría. Ejecutar con: pytest -q"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

# Ejemplo exacto del contrato (README del starter).
SAMPLE = {
    "robot_id": "R03", "timestamp": 1780000000.0, "x": 4.21, "y": 7.82, "speed": 0.63,
    "battery": 71.0, "motor_temp": 48.2, "vibration": 0.43, "motor_current": 3.7,
    "obstacle_distance": 0.42,
}
CONTRACT_KEYS = {"robot_id", "predicted_status", "confidence", "model_version", "inference_ms"}


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(f"sqlite:///{tmp_path / 't.db'}")) as c:
        yield c


def test_predict_contract(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == CONTRACT_KEYS
    assert body["robot_id"] == "R03"
    assert body["predicted_status"] in {"NORMAL", "WARNING", "CRITICAL"}
    assert isinstance(body["confidence"], float) and 0 <= body["confidence"] <= 1
    assert isinstance(body["model_version"], str) and body["model_version"]
    assert isinstance(body["inference_ms"], float) and body["inference_ms"] >= 0


def test_predict_clear_cases(client):
    healthy = {**SAMPLE, "motor_temp": 45.0, "vibration": 0.3, "motor_current": 3.8, "speed": 0.9}
    broken = {**SAMPLE, "motor_temp": 80.0, "vibration": 1.8, "motor_current": 6.0, "speed": 0.6}
    assert client.post("/predict", json=healthy).json()["predicted_status"] == "NORMAL"
    assert client.post("/predict", json=broken).json()["predicted_status"] == "CRITICAL"


def test_predict_does_not_store(client):
    client.post("/predict", json=SAMPLE)
    assert client.get("/robots").json() == []


@pytest.mark.parametrize("payload", [
    {**SAMPLE, "vibration": float("nan")},
    {**SAMPLE, "motor_temp": float("inf")},
    {k: v for k, v in SAMPLE.items() if k != "speed"},
    {**SAMPLE, "battery": 120},
    {**SAMPLE, "motor_current": "mucho"},
])
def test_predict_invalid_is_4xx(client, payload):
    r = client.post("/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"})
    assert 400 <= r.status_code < 500
    assert client.post("/predict", json=SAMPLE).status_code == 200


def test_predict_concurrent_requests(client):
    def call(i):
        return client.post("/predict", json={**SAMPLE, "robot_id": f"C{i:02d}"}).status_code
    with ThreadPoolExecutor(max_workers=20) as pool:
        codes = list(pool.map(call, range(200)))
    assert codes.count(200) == 200


def test_telemetry_stores_prediction(client):
    r = client.post("/telemetry", json=SAMPLE)
    assert r.json()["predicted_status"] in {"NORMAL", "WARNING", "CRITICAL"}
    robot = client.get("/robots").json()[0]
    assert robot["predicted_status"] == r.json()["predicted_status"]
    assert robot["model_version"] == client.post("/predict", json=SAMPLE).json()["model_version"]
    assert client.get("/robots/R03/history").json()[0]["predicted_status"] == robot["predicted_status"]


def test_health_reports_model(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True and body["model_version"]


def test_without_model_predict_is_503_and_telemetry_still_works(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'n.db'}", model_path=str(tmp_path / "missing.joblib"))
    with TestClient(app) as c:
        assert c.post("/predict", json=SAMPLE).status_code == 503
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/telemetry", json=SAMPLE).status_code == 200
        assert c.get("/robots").json()[0]["predicted_status"] is None


def test_fast_path_matches_full_pipeline():
    """El atajo NumPy debe dar las mismas probabilidades que el pipeline guardado."""
    import random
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from app.model import ModelService

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulator"))
    from core import make_robot, step_robot

    m = ModelService()
    m.load()
    assert m.info()["fast_path"] is True
    rng = random.Random(1)
    robots = [make_robot(i + 1, rng) for i in range(8)]
    for i in range(800):
        sample, _ = step_robot(robots[i % 8], rng)
        full = m.pipeline.predict_proba(pd.DataFrame([sample]))[0]
        assert np.allclose(m._fast(sample), full, atol=1e-12)
        assert m.predict(sample)["predicted_status"] == m.pipeline.classes_[full.argmax()]
