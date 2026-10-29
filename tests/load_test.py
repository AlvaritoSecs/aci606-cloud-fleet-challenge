"""Prueba de carga de POST /predict medida desde el CLIENTE, como lo hará el profesor.

Envía muestras realistas (generadas con simulator/core.py) desde varios hilos a la vez y
reporta errores, P50, P95 y máximo de la latencia completa (red + servidor + modelo).

Uso:
  python tests/load_test.py --api https://<servicio>.onrender.com --clients 20 --requests 400
  python tests/load_test.py --api ... --out reports/load_test_render.json

Antes de medir en Render, despierta el servicio (GET /health) para no medir el arranque en frío.
La latencia depende de dónde se ejecuta el cliente: anota siempre fecha, lugar y red.
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulator"))
from core import make_robot, step_robot  # noqa: E402

KEYS = {"robot_id", "predicted_status", "confidence", "model_version", "inference_ms"}


def make_samples(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    robots = [make_robot(i + 1, rng) for i in range(10)]
    samples = []
    for i in range(n):
        sample, _ = step_robot(robots[i % len(robots)], rng)
        samples.append({**sample, "robot_id": f"LOAD-{sample['robot_id']}"})
    return samples


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--clients", type=int, default=20, help="hilos concurrentes")
    p.add_argument("--requests", type=int, default=400, help="total de solicitudes")
    p.add_argument("--warmup", type=int, default=5, help="solicitudes previas que no se miden")
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", help="guardar el resultado en JSON")
    args = p.parse_args()
    url = args.api.rstrip("/") + "/predict"

    local = threading.local()

    def session() -> requests.Session:
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s

    samples = make_samples(args.requests + args.warmup, args.seed)
    print(f"Despertando el servicio ({args.warmup} solicitudes sin medir)...")
    for s in samples[: args.warmup]:
        requests.post(url, json=s, timeout=max(args.timeout, 90))

    def call(sample: dict) -> tuple[float, int, str | None]:
        t0 = time.perf_counter()
        try:
            r = session().post(url, json=sample, timeout=args.timeout)
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                return ms, r.status_code, r.text[:120]
            missing = KEYS - r.json().keys()
            return ms, 200, f"faltan {sorted(missing)}" if missing else None
        except requests.RequestException as exc:
            return (time.perf_counter() - t0) * 1000, 0, type(exc).__name__

    print(f"Enviando {args.requests} solicitudes a {url} con {args.clients} clientes concurrentes...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.clients) as pool:
        results = list(pool.map(call, samples[args.warmup:]))
    elapsed = time.perf_counter() - t0

    ok = [ms for ms, code, err in results if code == 200 and err is None]
    errors = [(code, err) for ms, code, err in results if code != 200 or err is not None]
    summary = {
        "api": args.api,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "client_host": socket.gethostname(),
        "clients": args.clients,
        "requests": args.requests,
        "ok": len(ok),
        "errors": len(errors),
        "error_samples": [f"{c} {e}" for c, e in errors[:5]],
        "throughput_rps": round(len(results) / elapsed, 1),
        "latency_ms": None if not ok else {
            "p50": round(percentile(ok, 50), 1),
            "p95": round(percentile(ok, 95), 1),
            "mean": round(statistics.fmean(ok), 1),
            "max": round(max(ok), 1),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Guardado en {args.out}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
