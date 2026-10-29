# Verificación del despliegue en Render

Pruebas de la lámina 15 del PDF ejecutadas contra la URL pública **https://aci606-cloud-fleet.onrender.com** el **29-10-2026** (hora UTC en cada salida), desde GitHub Codespaces. Las salidas están copiadas tal como aparecieron en la terminal.

| Verificación | Comando | Resultado |
|---|---|---|
| Smoke test final | `python tests/smoke_test.py --api <URL> --final` | **OK**: 13/13 comprobaciones; datos inválidos → 422 |
| Latencia de `/predict`, 1 cliente | `python tests/load_test.py --api <URL> --clients 1 --requests 100` | **0 errores**; P50 25,0 ms; P95 39,8 ms |
| Carga concurrente en `/predict`, 20 clientes | `python tests/load_test.py --api <URL> --clients 20 --requests 400` | **0 errores** en 400 solicitudes; P50 194,4 ms; P95 310,6 ms |
| Reinicio en vivo | `python tests/verify_restart.py --api <URL>` | **OK**: 5 robots con 184 registros antes y después |

## 1. Smoke test (`--final`)

```
GET /health                        -> 200
POST /telemetry                    -> 200
GET /robots                        -> 200
GET /metrics                       -> 200
POST /predict (contrato)           -> 200 {'robot_id': 'TEST-01', 'predicted_status': 'NORMAL', 'confidence': 0.9999, 'model_version': 'v1', 'inference_ms': 0.4}
GET /robots/TEST-01/history        -> 200 (1 registros)
GET /dashboard                     -> 200
POST /telemetry NaN                -> 422 (4xx esperado)
POST /telemetry campo faltante     -> 422 (4xx esperado)
POST /telemetry fuera de rango     -> 422 (4xx esperado)
POST /predict NaN                  -> 422 (4xx esperado)
POST /predict campo faltante       -> 422 (4xx esperado)
POST /predict fuera de rango       -> 422 (4xx esperado)

RESULTADO: OK
```

## 2. Latencia y concurrencia de `/predict` (medida desde el cliente)

Archivos: [`load_test_render_1client.json`](load_test_render_1client.json) y [`load_test_render.json`](load_test_render.json). Cliente: GitHub Codespaces. La latencia incluye la red entre Codespaces y Render, el servidor y el modelo; `inference_ms` (solo el modelo) fue 0,4 ms en el smoke test.

Un cliente (solicitudes una tras otra), 04:06:34 UTC:

```
"clients": 1, "requests": 100, "ok": 100, "errors": 0,
"throughput_rps": 35.4,
"latency_ms": { "p50": 25.0, "p95": 39.8, "mean": 28.2, "max": 88.8 }
```

Veinte clientes simultáneos, 04:00:20 UTC:

```
"clients": 20, "requests": 400, "ok": 400, "errors": 0,
"throughput_rps": 102.1,
"latency_ms": { "p50": 194.4, "p95": 310.6, "mean": 190.1, "max": 916.9 }
```

Contexto: el plan gratuito de Render es una instancia pequeña. Con 20 clientes simultáneos, gran parte del tiempo es espera en cola dentro del servidor. En un equipo local, el mismo código midió P50 32 ms y P95 38 ms con 20 clientes, y P50 2,9 ms con un cliente (ver el PR #4). La latencia que mida el profesor dependerá de su red y de su ubicación.

## 3. Reinicio en vivo e historial

Antes de la prueba, el simulador envió 150 pasos con 5 robots. El reinicio se pidió desde el panel de Render (Manual Deploy). El script detectó la nueva instancia porque cambió `started_at` en `/metrics`.

```
[04:00:38] Leyendo el historial antes del reinicio...
  R01              184 registros
  R02              184 registros
  R03              184 registros
  R04              184 registros
  R05              184 registros
  TEST-01            1 registros
  servicio encendido desde: 2026-10-29T03:56:22+00:00
[04:00:47] Esperando a que el servicio vuelva...
[04:03:29] Servicio reiniciado. Nuevo inicio: 2026-10-29T04:03:15+00:00

  robot          antes  después
  R01              184      184  OK
  R02              184      184  OK
  R03              184      184  OK
  R04              184      184  OK
  R05              184      184  OK
  TEST-01            1        1  OK

Robots con más de 100 registros: 5
RESULTADO: OK — el historial sobrevivió al reinicio
```

## 4. Panel desde otro dispositivo

El grupo abrió `/dashboard` con la flota en ejecución y confirmó que muestra los robots, sus estados predichos y el estado del servicio.

## 5. Revisión de secretos

Se buscaron cadenas de conexión, hosts de Neon y contraseñas en todo el historial con `git log -p --all`, sin resultados. `DATABASE_URL` solo existe como variable de entorno en Render.
