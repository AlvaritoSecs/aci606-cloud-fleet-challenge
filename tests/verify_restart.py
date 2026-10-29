"""Verifica que el historial sobrevive a un reinicio del servicio (lámina 15 del PDF).

Pasos que hace:
  1. Cuenta los registros del historial de cada robot.
  2. Te pide reiniciar el servicio en Render (Manual Deploy / Restart) y presionar Enter.
  3. Espera a que el servicio vuelva (detecta el reinicio porque cambia `started_at` en /metrics).
  4. Vuelve a contar y compara: ningún robot debe tener MENOS registros que antes.

Uso:
  python tests/verify_restart.py --api https://<servicio>.onrender.com

Sirve también para la demostración en vivo de la defensa.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import requests


def get(url: str, timeout: float) -> requests.Response:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def snapshot(base: str, timeout: float) -> dict[str, int]:
    robots = get(base + "/robots", timeout).json()
    counts = {}
    for r in robots:
        rid = r["robot_id"]
        counts[rid] = len(get(f"{base}/robots/{requests.utils.quote(rid, safe='')}/history", timeout).json())
    return counts


def started_at(base: str, timeout: float) -> str | None:
    try:
        return get(base + "/metrics", timeout).json().get("started_at")
    except requests.RequestException:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", required=True)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--wait", type=float, default=600, help="segundos máximos esperando que el servicio vuelva")
    args = p.parse_args()
    base = args.api.rstrip("/")

    print(f"[{datetime.now():%H:%M:%S}] Leyendo el historial antes del reinicio...")
    before = snapshot(base, args.timeout)
    first_start = started_at(base, args.timeout)
    if not before:
        sys.exit("No hay robots con historial. Ejecuta primero el simulador.")
    for rid, n in sorted(before.items()):
        print(f"  {rid:<12} {n:>7} registros")
    print(f"  servicio encendido desde: {first_start}")

    input("\nAhora REINICIA el servicio en Render (Manual Deploy -> Restart service, o Deploy latest commit)\n"
          "y presiona Enter aquí apenas lo hayas pedido...")

    print(f"[{datetime.now():%H:%M:%S}] Esperando a que el servicio vuelva", end="", flush=True)
    deadline = time.time() + args.wait
    new_start = None
    while time.time() < deadline:
        new_start = started_at(base, 15)
        if new_start and new_start != first_start:
            break
        print(".", end="", flush=True)
        time.sleep(5)
    print()
    if not new_start or new_start == first_start:
        sys.exit("El servicio no se reinició dentro del tiempo de espera (started_at no cambió).")
    print(f"[{datetime.now():%H:%M:%S}] Servicio reiniciado. Nuevo inicio: {new_start}")

    after = snapshot(base, args.timeout)
    ok = True
    print(f"\n  {'robot':<12} {'antes':>7} {'después':>8}")
    for rid in sorted(before):
        a, b = before[rid], after.get(rid, 0)
        flag = "OK" if b >= a else "SE PERDIERON DATOS"
        ok &= b >= a
        print(f"  {rid:<12} {a:>7} {b:>8}  {flag}")
    over_100 = [rid for rid, n in after.items() if n > 100]
    print(f"\nRobots con más de 100 registros: {len(over_100)}")
    print("RESULTADO:", "OK — el historial sobrevivió al reinicio" if ok else "FALLA")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
