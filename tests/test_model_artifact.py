"""Pruebas del artefacto del modelo (Fase 2). Ejecutar con: pytest -q"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
import sklearn

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "model.joblib"
META = ROOT / "models" / "model_meta.json"
CLASSES = {"NORMAL", "WARNING", "CRITICAL"}

# Una muestra con la forma exacta del contrato de /predict (README del starter).
SAMPLE = {
    "robot_id": "R03", "timestamp": 1780000000.0, "x": 4.21, "y": 7.82, "speed": 0.63,
    "battery": 71.0, "motor_temp": 48.2, "vibration": 0.43, "motor_current": 3.7,
    "obstacle_distance": 0.42,
}


@pytest.fixture(scope="module")
def model():
    if not MODEL.exists():
        pytest.skip("models/model.joblib no existe; ejecuta ml/train_model.py")
    return joblib.load(MODEL)


@pytest.fixture(scope="module")
def meta():
    return json.loads(META.read_text(encoding="utf-8"))


def test_meta_has_contract_fields(meta):
    for key in ("model_version", "features", "classes", "sklearn", "test_f1_macro"):
        assert key in meta
    assert set(meta["classes"]) == CLASSES


def test_sklearn_version_matches_training(meta):
    # Un pickle cargado con otra versión de scikit-learn puede fallar o predecir distinto.
    assert sklearn.__version__ == meta["sklearn"], (
        f"scikit-learn {sklearn.__version__} instalado, modelo entrenado con {meta['sklearn']}")


def test_predicts_valid_class_from_contract_payload(model):
    row = pd.DataFrame([SAMPLE])
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    assert pred in CLASSES
    assert abs(proba.sum() - 1) < 1e-9
    assert 0 <= proba.max() <= 1


def test_ignores_non_feature_fields(model):
    # El pipeline selecciona sus columnas: robot_id, x, y, timestamp no deben cambiar la predicción.
    a = pd.DataFrame([SAMPLE])
    b = pd.DataFrame([{**SAMPLE, "robot_id": "NUEVO-99", "x": 0.0, "y": 0.0, "timestamp": 1.0}])
    assert (model.predict_proba(a) == model.predict_proba(b)).all()


def test_obvious_cases(model):
    healthy = {**SAMPLE, "motor_temp": 45.0, "vibration": 0.3, "motor_current": 3.8, "speed": 0.9}
    broken = {**SAMPLE, "motor_temp": 80.0, "vibration": 1.8, "motor_current": 6.0, "speed": 0.6}
    assert model.predict(pd.DataFrame([healthy]))[0] == "NORMAL"
    assert model.predict(pd.DataFrame([broken]))[0] == "CRITICAL"
