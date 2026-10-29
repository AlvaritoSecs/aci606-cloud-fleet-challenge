"""Métricas operacionales en memoria para GET /metrics.

Un middleware registra cada solicitud: ruta, código HTTP y duración. Los contadores viven
en memoria y vuelven a cero cuando el servicio se reinicia; los DATOS no se pierden porque
están en PostgreSQL. `started_at` y `uptime_seconds` muestran desde cuándo se cuenta.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone

import numpy as np

WINDOW = 1000  # latencias guardadas para calcular percentiles (las más recientes)


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    arr = np.asarray(values)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "max": round(float(arr.max()), 3),
    }


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_monotonic = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.by_route: Counter[tuple[str, int]] = Counter()
        self.predict_ms: deque[float] = deque(maxlen=WINDOW)    # duración total de POST /predict en el servidor
        self.inference_ms: deque[float] = deque(maxlen=WINDOW)  # solo el modelo
        self.predicted_classes: Counter[str] = Counter()

    def record_request(self, route: str, status: int, duration_ms: float) -> None:
        with self._lock:
            self.by_route[(route, status)] += 1
            if route == "POST /predict" and status == 200:
                self.predict_ms.append(duration_ms)

    def record_prediction(self, predicted_status: str, inference_ms: float) -> None:
        with self._lock:
            self.inference_ms.append(inference_ms)
            self.predicted_classes[predicted_status] += 1

    def _count(self, route: str | None = None, lo: int = 0, hi: int = 1000) -> int:
        return sum(n for (r, s), n in self.by_route.items() if (route is None or r == route) and lo <= s < hi)

    def snapshot(self) -> dict:
        with self._lock:
            routes: dict[str, dict[str, int]] = {}
            for (route, status), n in sorted(self.by_route.items()):
                routes.setdefault(route, {})[str(status)] = n
            predict = _summary(list(self.predict_ms))
            return {
                "started_at": self.started_at,
                "uptime_seconds": round(time.monotonic() - self.started_monotonic, 1),
                "requests_total": self._count(),
                "telemetry_requests": self._count("POST /telemetry"),
                "telemetry_rejected_4xx": self._count("POST /telemetry", 400, 500),
                "prediction_requests": self._count("POST /predict"),
                "prediction_rejected_4xx": self._count("POST /predict", 400, 500),
                "errors_4xx": self._count(None, 400, 500),
                "errors_5xx": self._count(None, 500, 600),
                "avg_prediction_latency_ms": predict["mean"],
                "predict_latency_ms": {**predict, "window": WINDOW,
                                       "measured": "en el servidor, solicitud completa de POST /predict"},
                "inference_ms": {**_summary(list(self.inference_ms)), "measured": "solo el modelo"},
                "predicted_classes": dict(self.predicted_classes),
                "requests_by_route": routes,
            }
