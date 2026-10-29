"""Carga del modelo entrenado y predicción de una muestra.

El modelo (models/model.joblib) es el pipeline completo guardado por ml/train_model.py:
selección de columnas + escalado + clasificador. Se carga UNA vez al arrancar el servicio
y queda en memoria, así /predict no lee disco ni base de datos.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn

log = logging.getLogger("aci606.model")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"


class ModelService:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH)
        if not self.path.is_absolute():
            self.path = ROOT / self.path
        self.pipeline = None
        self.meta: dict = {}
        self.error: str | None = None
        self.version_warning: str | None = None

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    @property
    def version(self) -> str:
        return str(self.meta.get("model_version", "unknown"))

    def load(self) -> None:
        try:
            self.pipeline = joblib.load(self.path)
            meta_path = self.path.with_name("model_meta.json")
            if meta_path.exists():
                self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            trained_with = self.meta.get("sklearn")
            if trained_with and trained_with != sklearn.__version__:
                self.version_warning = (
                    f"Modelo entrenado con scikit-learn {trained_with}, instalado {sklearn.__version__}")
                log.warning(self.version_warning)
            self.error = None
            log.info("Modelo %s cargado desde %s", self.version, self.path)
        except Exception as exc:  # el servicio sigue vivo sin modelo: /predict responde 503
            self.pipeline = None
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("No se pudo cargar el modelo desde %s", self.path)

    def predict(self, sample: dict) -> dict:
        """Devuelve predicted_status, confidence y inference_ms (solo el tiempo del modelo)."""
        t0 = time.perf_counter()
        proba = self.pipeline.predict_proba(pd.DataFrame([sample]))[0]
        best = int(proba.argmax())
        inference_ms = (time.perf_counter() - t0) * 1000
        return {
            "predicted_status": str(self.pipeline.classes_[best]),
            "confidence": round(float(proba[best]), 4),
            "inference_ms": round(inference_ms, 3),
        }

    def info(self) -> dict:
        return {
            "loaded": self.ready,
            "version": self.version if self.ready else None,
            "name": self.meta.get("model_name"),
            "features": self.meta.get("features"),
            "test_f1_macro": self.meta.get("test_f1_macro"),
            "error": self.error,
            "warning": self.version_warning,
        }
