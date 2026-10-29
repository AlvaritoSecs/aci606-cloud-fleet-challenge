# ACI606 · Cloud Fleet Challenge — Ejercicio 4: Despliegue en PaaS

Servicio Cloud persistente que recibe la telemetría de una flota virtual de robots, la guarda en PostgreSQL, predice el estado de cada robot (NORMAL, WARNING o CRITICAL) con un modelo propio, expone métricas operacionales y muestra la flota en un panel web.

| | |
|---|---|
| **URL pública** | **https://aci606-cloud-fleet.onrender.com** |
| Panel | https://aci606-cloud-fleet.onrender.com/dashboard |
| Documentación interactiva de la API | https://aci606-cloud-fleet.onrender.com/docs |
| Plataforma (PaaS) | Render: Web Service, plan Free, despliegue automático desde `main` |
| Datos | Neon: PostgreSQL gestionado, región AWS us-east-1 (Virginia), tabla `telemetry` |
| Código | GitHub: este repositorio |

> **Primera solicitud lenta:** el plan gratuito de Render suspende el servicio tras 15 minutos sin tráfico y tarda alrededor de un minuto en despertar. Antes de evaluar, abre `/health` y espera la respuesta.

El enunciado original del profesor está en [`docs/STARTER_README.md`](docs/STARTER_README.md).

---

## 1. Arquitectura

```text
  Simulador de flota · Navegador (/dashboard) · Evaluador
                    │  HTTPS
                    ▼
  ┌───────────────────── Render · Web Service ─────────────────────┐
  │ FastAPI + Uvicorn (0.0.0.0:$PORT)                               │
  │  · Validación: campos, tipos, números finitos y rangos → 422    │
  │  · Modelo en memoria: models/model.joblib, cargado al arrancar  │
  │  · Middleware de métricas: solicitudes, errores, latencias      │
  │  · Caché de la última muestra por robot (se reconstruye de la   │
  │    base al arrancar)                                            │
  └───────────────┬─────────────────────────────────────────────────┘
                  │ SQL con TLS (SQLAlchemy + psycopg 3, pool_pre_ping)
                  ▼
  ┌──────────── Neon · PostgreSQL ────────────┐        GitHub (main)
  │ tabla telemetry: historial completo,      │   ──── push → deploy
  │ con la predicción de cada muestra         │        automático en Render
  └───────────────────────────────────────────┘
```

**Recorrido de una muestra:**
1. El simulador envía `POST /telemetry`.
2. FastAPI la valida. Si es inválida responde 422 y no guarda nada.
3. Si es válida, el modelo en memoria calcula la predicción en menos de 1 ms.
4. Se inserta una fila en Neon con la muestra y su predicción.
5. Recién después de guardarla en la base, se actualiza la caché de la última muestra del robot.
6. El panel consulta `/robots`, `/metrics`, `/health` y el historial del robot que se elija.

**Decisiones de diseño:**

| Decisión | Por qué |
|---|---|
| PostgreSQL en Neon, no SQLite ni archivos | El disco de Render es efímero: se borra en cada reinicio o redeploy. Neon es independiente del servicio y no expira, a diferencia de la base gratuita de Render, que se borra a los 30 días. |
| `DATABASE_URL` solo como variable de entorno | No hay credenciales en el código ni en Git. Si la variable falta, el servicio no arranca: falla de forma visible en vez de guardar datos donde se pierden. |
| `pool_pre_ping=True` y `pool_recycle=300` | Neon suspende la base tras 5 minutos sin uso y corta las conexiones. Cada conexión se prueba antes de usarse y se reemplaza si está muerta. |
| `/predict` no toca la base | Su latencia no depende de la red ni de que Neon esté despierta. |
| La predicción se guarda con cada telemetría | El panel y el historial muestran el estado predicho sin volver a llamar al modelo. |
| `/health` no consulta la base | Render llama `/health` seguido; consultar la base en cada llamada mantendría a Neon despierta y gastaría su cuota mensual de cómputo. |
| Modelo versionado en Git (`git add -f`) | Es una de las dos opciones del PDF. Es más simple y predecible que entrenar durante el build. |
| Versiones fijas (`requirements.txt`, `.python-version` = 3.13) | Un modelo de scikit-learn cargado con otra versión puede fallar o predecir distinto. |

---

## 2. API

| Ruta | Qué hace |
|---|---|
| `GET /health` | Estado del servicio, de la base y del modelo. No consulta la base. |
| `POST /telemetry` | Valida la muestra, predice su estado y la guarda. Responde `accepted`, `robot_id`, `server_time`, `note` y `predicted_status`. |
| `GET /robots` | Última muestra de cada robot, con su predicción. |
| `GET /robots/{robot_id}/history` | Historial **completo** del robot en orden de llegada; 404 si no existe. El parámetro opcional `?limit=N` devuelve solo los últimos N registros (lo usa el panel). |
| `POST /predict` | Predicción sin guardar nada. Responde `robot_id`, `predicted_status`, `confidence`, `model_version` e `inference_ms`. |
| `GET /metrics` | Métricas operacionales (sección 5). |
| `GET /dashboard` | Panel web. |

Ejemplo:

```bash
curl -X POST https://aci606-cloud-fleet.onrender.com/predict -H "Content-Type: application/json" \
  -d '{"robot_id":"R03","timestamp":1780000000.0,"x":4.21,"y":7.82,"speed":0.63,"battery":71.0,"motor_temp":48.2,"vibration":0.43,"motor_current":3.7,"obstacle_distance":0.42}'
# {"robot_id":"R03","predicted_status":"NORMAL","confidence":0.9999,"model_version":"v1","inference_ms":0.4}
```

### Validación de entradas

Las entradas inválidas reciben 422 y nunca 500, y el servicio sigue sano después. Se considera inválido:
- un campo faltante;
- un tipo incorrecto;
- NaN o Infinity en cualquier número;
- un valor fuera de rango.

Los rangos son **amplios a propósito**: la flota de evaluación puede tener "condiciones levemente distintas" y no se rechazan lecturas físicamente plausibles. Las 141 500 muestras de entrenamiento y evaluación generadas por el simulador son aceptadas.

| Campo | Permitido | Unidad | Rango típico del simulador |
|---|---|---|---|
| `robot_id` | texto de 1 a 20 caracteres | — | R01, R02, … |
| `timestamp` | 0 a 1e13 | s desde 1970 | `time.time()` |
| `x`, `y` | −1000 a 1000 | m | 0 a 10 |
| `speed` | 0 a 20 | m/s | 0,05 a 1,25 |
| `battery` | 0 a 100 | % | 0 a 100 |
| `motor_temp` | −40 a 200 | °C | aprox. 35 a 90 |
| `vibration` | 0 a 50 | unidades del simulador | aprox. 0 a 2 |
| `motor_current` | 0 a 100 | A | aprox. 0,2 a 8 |
| `obstacle_distance` | 0 a 1000 | m | 0,05 a 4 |

Un detalle verificado: el manejador de errores 422 que trae FastAPI devuelve el valor recibido, y un NaN no se puede escribir como JSON, así que ese 422 terminaba como 500. Por eso `app/main.py` usa un manejador propio.

---

## 3. Modelo de Machine Learning

**Resultado: regresión logística con 4 variables (`speed`, `motor_temp`, `vibration`, `motor_current`). En 10 flotas nunca vistas obtiene F1 macro 0,923, contra 0,297 de la línea base "siempre CRITICAL".**

| Paso | Script | Evidencia |
|---|---|---|
| Dataset multiflota con `fleet_id` | `ml/build_dataset.py` | 30 flotas de entrenamiento y 10 reservadas |
| Análisis exploratorio | `ml/eda.py` | [`reports/eda.md`](reports/eda.md) y [`reports/eda/`](reports/eda/) |
| Comparación y selección | `ml/train_model.py` | [`reports/model_comparison.md`](reports/model_comparison.md) y [matriz de confusión](reports/confusion_matrix_test.png) |
| Evaluación en flotas no vistas | `ml/evaluate_unseen.py` | [`reports/unseen_evaluation.md`](reports/unseen_evaluation.md) |
| Pipeline guardado | — | `models/model.joblib` y `models/model_meta.json` |

**Datos.** El generador del profesor no produce `fleet_id`, que se necesita para separar por flota. `build_dataset.py` llama al generador **sin modificarlo** una vez por semilla (una semilla = una flota), con largos de 1 500 a 8 000 muestras y de 4 a 12 robots, y agrega `fleet_id`.

**Hallazgos del EDA:**
- CRITICAL es la clase **mayoritaria** (74 %). Por eso la métrica principal es F1 macro y los modelos usan `class_weight="balanced"`.
- Los robots se degradan con el tiempo: en el primer cuarto de cada simulación hay 29 % de NORMAL, y en el último cuarto, 2 %.
- `battery` se correlaciona con el tiempo (−0,92) más que con el estado: baja por uso, no por falla.
- `obstacle_distance` no se relaciona con la clase (0,001).

**Protocolo sin fuga de datos:**
- `GroupShuffleSplit` por `fleet_id`: 22 flotas para entrenar y 8 para probar, sin flotas compartidas.
- La elección de modelo se hace con `GroupKFold(5)` solo sobre las flotas de entrenamiento.
- La evaluación final usa 10 flotas con semillas reservadas desde el inicio.
- `robot_id`, `timestamp`, `x` e `y` no son variables del modelo.

**Comparación** (F1 macro; validación cruzada por flota y flotas de prueba):

| Modelo | Variables | CV (media ± desv.) | Prueba |
|---|---|---|---|
| Línea base "siempre CRITICAL" | — | 0,282 ± 0,015 | 0,287 |
| Regresión logística | 6 | 0,923 ± 0,011 | 0,933 |
| **Regresión logística** | **4** (sin `battery` ni `obstacle_distance`) | **0,922 ± 0,011** | **0,932** |
| HistGradientBoosting | 6 | 0,922 ± 0,010 | 0,930 |
| Random Forest | 6 | 0,920 ± 0,012 | 0,929 |

**Justificación.** Los tres clasificadores quedan dentro de una desviación estándar entre sí. La regla de selección, fijada antes de ver los resultados, elige entonces el modelo más simple y con menos variables. La regresión logística:
- es la más rápida (0,4 ms por predicción en Render);
- es explicable con sus coeficientes;
- no depende de `battery`, cuya relación con el estado puede cambiar si la flota de evaluación dura más o menos tiempo.

El F1 no llega a 1 por un límite del problema: la etiqueta depende de una "salud" oculta con umbrales (0,38 y 0,62), y cerca de esos umbrales los sensores no distinguen las clases. Los errores son casi todos entre clases vecinas, por ejemplo WARNING confundido con NORMAL o con CRITICAL.

Para reproducirlo:

```bash
pip install -r requirements-dev.txt
python ml/build_dataset.py && python ml/eda.py && python ml/train_model.py && python ml/evaluate_unseen.py
```

Dos ejecuciones dan exactamente los mismos resultados (semillas fijas).

---

## 4. Despliegue

### Procedimiento en Render + Neon

1. **Neon:** crear un proyecto en la región AWS us-east-1 (Virginia). En **Connect**, copiar la cadena `postgresql://…` y guardarla solo en un lugar privado.
2. **Render:** ir a **New → Web Service** y conectar este repositorio, rama `main`, plan **Free**.
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Health check: `/health`
3. **Render → Environment:** agregar `DATABASE_URL` con la cadena de Neon. El código agrega solo el driver `psycopg`. `MODEL_PATH` es opcional; por defecto es `models/model.joblib`.
4. Cada merge a `main` dispara un deploy automático. La tabla `telemetry` se crea sola al arrancar.

Python se fija con `.python-version` (3.13). Sin ese archivo, Render usa 3.14.8 por defecto en servicios nuevos, y el modelo se entrenó con 3.13.

### Ejecución local

```bash
python -m venv .venv && source .venv/bin/activate       # En Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export DATABASE_URL=sqlite:///./local.db                 # o una base de prueba de Neon (ver .env.example)
uvicorn app.main:app --reload
# en otra terminal:
python simulator/fleet_simulator.py --api http://127.0.0.1:8000 --robots 5
```

### Verificación (lámina 15)

Las tres verificaciones se ejecutaron contra la URL pública el 29-10-2026. Las salidas completas están en [`reports/deploy_verification.md`](reports/deploy_verification.md).

| Verificación | Resultado |
|---|---|
| `python tests/smoke_test.py --api <URL> --final` | **OK**: 13/13; NaN, campo faltante y fuera de rango → 422 |
| `python tests/verify_restart.py --api <URL>` | **OK**: 5 robots con 184 registros antes y después del reinicio |
| `python tests/load_test.py --api <URL> --clients 1 --requests 100` | **0 errores**; P50 25,0 ms; P95 39,8 ms (desde GitHub Codespaces) |
| `python tests/load_test.py --api <URL> --clients 20 --requests 400` | **0 errores**; P50 194,4 ms; P95 310,6 ms con 20 clientes simultáneos |

---

## 5. Observabilidad

`GET /metrics` mantiene las claves del starter y agrega:

| Clave | Significado |
|---|---|
| `telemetry_requests`, `requests_received` | Telemetrías recibidas (todas, y solo las aceptadas) |
| `prediction_requests` | Llamadas a `/predict` |
| `errors_4xx`, `errors_5xx`, `telemetry_rejected_4xx`, `prediction_rejected_4xx` | Errores; 4xx = datos rechazados |
| `avg_prediction_latency_ms`, `predict_latency_ms` | Media, P50, P95 y máximo de `/predict` en el servidor (últimas 1000 llamadas) |
| `inference_ms` | Tiempo solo del modelo |
| `uptime_seconds`, `started_at` | Tiempo encendido y hora de arranque |
| `predicted_classes`, `requests_by_route` | Predicciones por clase; solicitudes por ruta y código HTTP |
| `database`, `model` | Estado de la base y del modelo cargado |

Los contadores viven en memoria y **vuelven a cero cuando el servicio se reinicia**; los datos no se pierden porque están en PostgreSQL. La latencia que mide el profesor desde su equipo incluye además la red: para medirla igual que él está `tests/load_test.py`.

**Rendimiento.** Al medir, se vio que de los ~3,7 ms por predicción, ~2,6 ms se iban en `ColumnTransformer` y en crear el `DataFrame`. Ahora `app/model.py` hace los mismos pasos con NumPy (mismo orden de columnas, mismo escalado, mismo clasificador). Ese atajo solo se activa si al cargar reproduce **exactamente** las probabilidades del pipeline, y un test lo verifica con 800 muestras. Además, `/predict` corre en el bucle de eventos. En un equipo local, con 20 clientes concurrentes, P50 bajó de 129 ms a 32 ms y el rendimiento subió de 144 a 590 solicitudes por segundo.

---

## 6. Panel

`/dashboard` muestra:
- robots conectados (los que enviaron telemetría en los últimos 15 s);
- telemetría más reciente y batería de cada robot;
- estado predicho con color y texto, y su confianza;
- conteo de la flota por estado;
- historial del robot elegido: temperatura, vibración y franja de estados de las últimas 150 muestras;
- estado de la API, la base y el modelo, con predicciones, P50/P95, errores y tiempo encendido.

Todo texto que viene de la API se escapa antes de mostrarse.

---

## 7. Pruebas

```bash
pytest                                             # 47 pruebas, con SQLite temporal
TEST_DATABASE_URL=postgresql://... pytest tests/test_api.py   # contra PostgreSQL de PRUEBA (borra la tabla)
```

Cubren: historial de más de 100 registros, persistencia tras reiniciar, `/robots` con la última muestra, 404, 14 casos inválidos en 4xx, contrato de `/predict`, 200 predicciones concurrentes, servicio sin modelo (503), métricas y el atajo NumPy idéntico al pipeline.

---

## 8. Trazabilidad Git

Cada fase se desarrolló en su propia rama y se integró a `main` mediante Pull Request con *merge commit* (`git log --graph --oneline`):

| PR | Rama | Contenido |
|---|---|---|
| #1 | `feature/ml-pipeline` | Dataset multiflota, EDA, comparación de modelos, `model.joblib` |
| #2 | `feature/persistence` | PostgreSQL, validación robusta, 503 ante fallas de la base |
| #3 | `feature/predict` | `/predict`, predicción guardada con la telemetría, `smoke_test --final` |
| #4 | `feature/metrics` | `/metrics`, prueba de carga, inferencia más rápida |
| #5 | `feature/dashboard` | Panel con predicciones, agregados e historial |
| #6 | `feature/verification` | Script de verificación de reinicio |
| #7 | `feature/docs` | Este README y las evidencias de despliegue |

No hay credenciales en el historial: se revisó con `git log -p --all`. `.env` está en `.gitignore` y `.env.example` solo trae los nombres de las variables.

---

## 9. Limitaciones conocidas

- **Plan gratuito:** Render suspende el servicio tras 15 minutos sin tráfico y tarda ~1 minuto en despertar; además, puede reiniciarlo sin aviso. Neon suspende la base tras 5 minutos y tiene una cuota de 100 CU-hora al mes. Ninguno garantiza disponibilidad.
- **Una sola instancia:** las métricas y la caché viven en memoria de ese proceso. Con varias instancias habría que mover las métricas a un sistema externo.
- **Sin autenticación:** cualquiera con la URL puede enviar telemetría. El enunciado no lo pide.
- **Historial sin paginación por defecto:** el contrato pide el historial completo. Con días de simulación continua la respuesta crece; `?limit=N` existe para el panel.
- **El modelo se evaluó con flotas del mismo simulador:** si la flota de evaluación cambia la relación entre sensores y estado, el F1 puede bajar.

## 10. Estructura

```text
app/        main.py (rutas, arranque), schemas.py (validación), db.py (PostgreSQL),
            model.py (modelo), metrics.py (métricas)
dashboard/  index.html (panel)
ml/         build_dataset.py, eda.py, train_model.py, evaluate_unseen.py, config.py
models/     model.joblib, model_meta.json
reports/    EDA, comparación de modelos, evaluación en flotas no vistas, verificación del despliegue
simulator/  simulador y generador del profesor (sin modificar)
tests/      pytest (test_*.py) y scripts contra un servidor real (smoke_test, load_test, verify_restart)
```
