"""Configuración compartida del pipeline de ML (Fase 2).

Todas las semillas y rutas están aquí para que el pipeline sea reproducible:
cualquier persona que ejecute los scripts obtiene los mismos datos y el mismo modelo.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TRAINING_CSV = DATA_DIR / "training.csv"   # flotas para entrenar y validar
UNSEEN_CSV = DATA_DIR / "unseen.csv"       # flotas reservadas: solo para la evaluación final

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
META_PATH = MODELS_DIR / "model_meta.json"

REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 606

# Campos que envía el simulador (contrato de la API).
TELEMETRY_FIELDS = [
    "robot_id", "timestamp", "x", "y", "speed", "battery",
    "motor_temp", "vibration", "motor_current", "obstacle_distance",
]

# Variables candidatas del starter (ml/train_model.py original).
# robot_id, timestamp, x e y quedan fuera a propósito: no describen el estado del robot.
FEATURES_ALL = ["speed", "battery", "motor_temp", "vibration", "motor_current", "obstacle_distance"]

TARGET = "status"
CLASSES = ["NORMAL", "WARNING", "CRITICAL"]
GROUP = "fleet_id"

# Flotas de entrenamiento: 30 semillas, combinando largos y tamaños distintos
# para cubrir flotas "jóvenes" (más NORMAL) y "viejas" (casi todo CRITICAL).
TRAIN_SEEDS = list(range(101, 131))
TRAIN_SAMPLES = [1500, 3000, 6000]
TRAIN_ROBOTS = [5, 8, 10]

# Flotas reservadas, NUNCA usadas para decidir nada: imitan la flota no vista
# del profesor con otras semillas, otros largos y otra cantidad de robots.
UNSEEN_SEEDS = list(range(9001, 9011))
UNSEEN_SAMPLES = [1000, 2500, 5000, 8000]
UNSEEN_ROBOTS = [4, 6, 12]
