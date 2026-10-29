"""Smoke test del servicio (basado en el del profesor).

Uso:
  python tests/smoke_test.py --api https://<servicio>.onrender.com
  python tests/smoke_test.py --api https://<servicio>.onrender.com --final

Sin --final hace las mismas 4 comprobaciones del starter. Con --final (comando de la
lámina 15 del PDF) agrega /predict con su contrato, el historial, el panel y el rechazo
de datos inválidos con 4xx. Si el profesor publica su propia versión de --final, usar esa.
"""
from __future__ import annotations

import argparse
import json
import time

import requests

SAMPLE = {
    "robot_id": "TEST-01",
    "timestamp": time.time(),
    "x": 1.2,
    "y": 3.4,
    "speed": 0.72,
    "battery": 81.0,
    "motor_temp": 49.0,
    "vibration": 0.42,
    "motor_current": 3.8,
    "obstacle_distance": 1.3,
}
PREDICT_KEYS = {"robot_id", "predicted_status", "confidence", "model_version", "inference_ms"}


def check_predict(response: requests.Response) -> str | None:
    if not response.ok:
        return f"HTTP {response.status_code}"
    body = response.json()
    if not PREDICT_KEYS <= body.keys():
        return f"faltan campos: {sorted(PREDICT_KEYS - body.keys())}"
    if body["predicted_status"] not in {"NORMAL", "WARNING", "CRITICAL"}:
        return f"predicted_status inválido: {body['predicted_status']}"
    if not 0 <= float(body["confidence"]) <= 1:
        return f"confidence fuera de 0..1: {body['confidence']}"
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--final", action="store_true", help="incluye /predict, historial, panel y datos inválidos")
    p.add_argument("--timeout", type=float, default=60, help="segundos; Render free puede tardar ~1 min en despertar")
    args = p.parse_args()
    base = args.api.rstrip("/")
    t = args.timeout

    checks = [
        ("GET /health", requests.get(base + "/health", timeout=t)),
        ("POST /telemetry", requests.post(base + "/telemetry", json=SAMPLE, timeout=t)),
        ("GET /robots", requests.get(base + "/robots", timeout=t)),
        ("GET /metrics", requests.get(base + "/metrics", timeout=t)),
    ]

    failed = False
    for name, response in checks:
        print(f"{name:<34} -> {response.status_code}")
        if not response.ok:
            failed = True

    if args.final:
        r = requests.post(base + "/predict", json=SAMPLE, timeout=t)
        problem = check_predict(r)
        print(f"{'POST /predict (contrato)':<34} -> {r.status_code} {problem or r.json()}")
        failed |= problem is not None

        r = requests.get(base + "/robots/TEST-01/history", timeout=t)
        print(f"{'GET /robots/TEST-01/history':<34} -> {r.status_code} ({len(r.json()) if r.ok else 0} registros)")
        failed |= not r.ok

        r = requests.get(base + "/dashboard", timeout=t)
        print(f"{'GET /dashboard':<34} -> {r.status_code}")
        failed |= not r.ok

        invalid = {
            "NaN": json.dumps({**SAMPLE, "vibration": float("nan")}),
            "campo faltante": json.dumps({k: v for k, v in SAMPLE.items() if k != "speed"}),
            "fuera de rango": json.dumps({**SAMPLE, "battery": 150}),
        }
        for route in ("/telemetry", "/predict"):
            for label, body in invalid.items():
                r = requests.post(base + route, data=body, headers={"Content-Type": "application/json"}, timeout=t)
                ok = 400 <= r.status_code < 500
                print(f"{'POST ' + route + ' ' + label:<34} -> {r.status_code} {'(4xx esperado)' if ok else 'FALLA'}")
                failed |= not ok

    print("\nRESULTADO:", "FALLA" if failed else "OK")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
