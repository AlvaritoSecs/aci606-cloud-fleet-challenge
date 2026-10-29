"""Contrato de entrada de la API y su validación.

Reglas:
- Los nombres de campo son los del starter y del simulador; no se cambian.
- Todo número debe ser finito: NaN e Infinity se rechazan con 422.
- Los rangos son AMPLIOS a propósito. El simulador produce valores mucho más acotados
  (ver simulator/core.py), pero la flota de evaluación puede venir con "condiciones
  levemente distintas" y no queremos rechazar datos físicamente plausibles. Solo se
  rechaza lo que no puede ser una lectura real (temperatura de -500 °C, velocidad de 999 m/s).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["NORMAL", "WARNING", "CRITICAL"]

# campo -> (mínimo, máximo, unidad, rango típico observado en el simulador)
RANGES: dict[str, tuple[float, float, str, str]] = {
    "timestamp": (0, 1e13, "s desde 1970", "time.time()"),
    "x": (-1_000, 1_000, "m", "0 a 10"),
    "y": (-1_000, 1_000, "m", "0 a 10"),
    "speed": (0, 20, "m/s", "0,05 a 1,25"),
    "battery": (0, 100, "%", "0 a 100"),
    "motor_temp": (-40, 200, "°C", "aprox. 35 a 90"),
    "vibration": (0, 50, "unidades del simulador", "aprox. 0 a 2"),
    "motor_current": (0, 100, "A", "aprox. 0,2 a 8"),
    "obstacle_distance": (0, 1_000, "m", "0,05 a 4"),
}


def _number(name: str):
    lo, hi, unit, _ = RANGES[name]
    return Field(ge=lo, le=hi, allow_inf_nan=False, description=f"{unit}; permitido {lo} a {hi}")


class Telemetry(BaseModel):
    """Una muestra de telemetría (POST /telemetry y POST /predict)."""

    # Campos extra se ignoran (comportamiento del starter) para no romper clientes.
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={"examples": [{
            "robot_id": "R03", "timestamp": 1780000000.0, "x": 4.21, "y": 7.82, "speed": 0.63,
            "battery": 71.0, "motor_temp": 48.2, "vibration": 0.43, "motor_current": 3.7,
            "obstacle_distance": 0.42,
        }]},
    )

    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float = _number("timestamp")
    x: float = _number("x")
    y: float = _number("y")
    speed: float = _number("speed")
    battery: float = _number("battery")
    motor_temp: float = _number("motor_temp")
    vibration: float = _number("vibration")
    motor_current: float = _number("motor_current")
    obstacle_distance: float = _number("obstacle_distance")


class PredictionRequest(Telemetry):
    pass


TELEMETRY_FIELDS = list(Telemetry.model_fields)
