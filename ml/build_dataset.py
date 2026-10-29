"""Construye los datasets multiflota usando el generador oficial SIN modificarlo.

El PDF exige separar entrenamiento y prueba por `fleet_id`, pero
`simulator/generate_dataset.py` no genera esa columna. Aquí se llama al generador
una vez por flota (una semilla = una flota) y se agrega `fleet_id`.

Salida:
  data/training.csv  flotas para entrenar y validar
  data/unseen.csv    flotas reservadas para la evaluación final

Uso:
  python ml/build_dataset.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    DATA_DIR, ROOT, TRAINING_CSV, TRAIN_ROBOTS, TRAIN_SAMPLES, TRAIN_SEEDS,
    UNSEEN_CSV, UNSEEN_ROBOTS, UNSEEN_SAMPLES, UNSEEN_SEEDS,
)

GENERATOR = ROOT / "simulator" / "generate_dataset.py"


def fleet_plan(seeds: list[int], samples: list[int], robots: list[int]) -> list[tuple[int, int, int]]:
    """Asigna a cada semilla un largo y un tamaño de flota, recorriendo todas las combinaciones."""
    plan = []
    for i, seed in enumerate(seeds):
        n_samples = samples[i % len(samples)]
        n_robots = robots[(i // len(samples)) % len(robots)]
        plan.append((seed, n_samples, n_robots))
    return plan


def generate_fleet(seed: int, n_samples: int, n_robots: int, workdir: Path) -> pd.DataFrame:
    out = workdir / f"fleet_{seed}.csv"
    subprocess.run(
        [sys.executable, str(GENERATOR),
         "--samples", str(n_samples), "--robots", str(n_robots),
         "--seed", str(seed), "--output", str(out)],
        check=True, capture_output=True, text=True,
    )
    df = pd.read_csv(out)
    df.insert(0, "fleet_id", seed)
    # El generador recorre los robots en orden: la fila i corresponde al paso i // n_robots.
    df["step"] = df.index // n_robots
    df["fleet_steps"] = n_samples // n_robots
    return df


def build(plan: list[tuple[int, int, int]], output: Path, label: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        frames = [generate_fleet(seed, s, r, Path(tmp)) for seed, s, r in plan]
    df = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    counts = df["status"].value_counts().reindex(["NORMAL", "WARNING", "CRITICAL"], fill_value=0)
    print(f"{label}: {output.relative_to(ROOT)} -> {len(df)} filas, {df['fleet_id'].nunique()} flotas")
    print("  clases:", counts.to_dict())
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train = build(fleet_plan(TRAIN_SEEDS, TRAIN_SAMPLES, TRAIN_ROBOTS), TRAINING_CSV, "Entrenamiento")
    unseen = build(fleet_plan(UNSEEN_SEEDS, UNSEEN_SAMPLES, UNSEEN_ROBOTS), UNSEEN_CSV, "No vistas")
    overlap = set(train["fleet_id"]) & set(unseen["fleet_id"])
    assert not overlap, f"Flotas compartidas entre conjuntos: {overlap}"


if __name__ == "__main__":
    main()
