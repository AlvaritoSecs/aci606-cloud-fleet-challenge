# EDA del dataset multiflota

Datos: `data/training.csv`, 105000 filas, 30 flotas, 6 variables candidatas. Valores faltantes: 0.

## 1. Distribución de clases

| clase | filas | porcentaje |
|---|---|---|
| NORMAL | 12239 | 11.7 |
| WARNING | 15120 | 14.4 |
| CRITICAL | 77641 | 73.9 |

Por flota, el porcentaje de CRITICAL va de 29.3 % a 95.1 % (mediana 69.3 %). CRITICAL es la clase mayoritaria: la línea base 'predecir siempre CRITICAL' tiene accuracy alta pero F1 macro bajo, por eso la métrica principal es F1 macro.

## 2. Clases según el avance de la simulación

Porcentaje de cada clase según el tramo de la vida de la flota:

| tramo | NORMAL | WARNING | CRITICAL |
|---|---|---|---|
| 0-25 % | 28.8 | 27.4 | 43.8 |
| 25-50 % | 11.5 | 16.3 | 72.2 |
| 50-75 % | 4.2 | 8.9 | 86.9 |
| 75-100 % | 2.2 | 5.0 | 92.9 |

Los robots se degradan con el tiempo (la salud oculta tiene caídas bruscas), así que una flota recién iniciada tiene muchas más muestras NORMAL que una flota antigua. La flota de evaluación puede tener otra duración: el modelo no debe depender del tiempo.

## 3. Media de cada variable por clase

| variable | NORMAL | WARNING | CRITICAL |
|---|---|---|---|
| speed | 0.854 | 0.811 | 0.743 |
| battery | 78.614 | 76.917 | 58.740 |
| motor_temp | 52.087 | 61.226 | 74.458 |
| vibration | 0.534 | 0.951 | 1.554 |
| motor_current | 4.445 | 4.880 | 5.492 |
| obstacle_distance | 1.313 | 1.320 | 1.317 |

## 4. Correlaciones

| variable | corr. con gravedad (0=NORMAL, 2=CRITICAL) | corr. con el paso de tiempo |
|---|---|---|
| speed | -0.311 | -0.161 |
| battery | -0.386 | -0.915 |
| motor_temp | 0.899 | 0.451 |
| vibration | 0.908 | 0.457 |
| motor_current | 0.642 | 0.318 |
| obstacle_distance | 0.001 | -0.004 |

Lectura: `motor_temp`, `vibration` y `motor_current` son las señales directas del deterioro. `battery` se correlaciona con la gravedad, pero sobre todo con el tiempo: baja por el uso y la flota se deteriora en paralelo, sin que la batería cause el estado. Esa relación puede no repetirse en una flota de otra duración, así que se evalúa el modelo con y sin `battery`. `obstacle_distance` casi no se relaciona con la clase.

## Gráficos

- `eda/class_distribution.png`
- `eda/class_by_time.png`
- `eda/features_by_class.png`
- `eda/correlation.png`
