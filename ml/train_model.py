"""Entrenamiento y selección del modelo (Ejercicio 4, Fase 2).

Protocolo:
  1. Separar entrenamiento / prueba POR FLOTA con GroupShuffleSplit (fleet_id),
     como exige el PDF. Ninguna flota aparece en ambos conjuntos.
  2. Elegir modelo y variables con GroupKFold (5 pliegues) solo sobre las flotas de
     entrenamiento: así la elección no mira el conjunto de prueba.
  3. Reportar F1 macro y matriz de confusión del modelo elegido en las flotas de prueba.
  4. Reentrenar el modelo elegido con todas las flotas etiquetadas y guardar el
     pipeline completo (selección de columnas + escalado + clasificador) en
     models/model.joblib, con sus metadatos en models/model_meta.json.
  La evaluación final en flotas reservadas está en ml/evaluate_unseen.py.

Uso:
  python ml/build_dataset.py
  python ml/train_model.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CLASSES, FEATURES_ALL, GROUP, META_PATH, MODEL_PATH, MODELS_DIR, RANDOM_STATE,
    REPORTS_DIR, ROOT, TARGET, TELEMETRY_FIELDS, TRAINING_CSV,
)

MODEL_VERSION = "v1"
TEST_SIZE = 0.25
CV_FOLDS = 5

FEATURE_SETS = {
    "completo": FEATURES_ALL,
    "sin_battery": [f for f in FEATURES_ALL if f != "battery"],
    "sin_battery_obstacle": [f for f in FEATURES_ALL if f not in ("battery", "obstacle_distance")],
}

# nombre -> (complejidad para desempatar, fábrica del clasificador, ¿necesita escalado?)
MODELS = {
    "logistic_regression": (1, lambda: LogisticRegression(max_iter=2000, class_weight="balanced"), True),
    "hist_gradient_boosting": (2, lambda: HistGradientBoostingClassifier(
        class_weight="balanced", random_state=RANDOM_STATE), False),
    "random_forest": (3, lambda: RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=5, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE), False),
}


def build_pipeline(model_name: str, features: list[str]) -> Pipeline:
    """El pipeline recibe un DataFrame con los campos de telemetría y selecciona sus columnas."""
    if model_name == "baseline_mayoritaria":
        prep, clf = "passthrough", DummyClassifier(strategy="most_frequent")
    else:
        _, factory, scale = MODELS[model_name]
        prep, clf = (StandardScaler() if scale else "passthrough"), factory()
    columns = ColumnTransformer([("num", prep, features)], remainder="drop",
                                verbose_feature_names_out=False)
    return Pipeline([("columns", columns), ("clf", clf)])


def md_table(rows: list[dict]) -> str:
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()) + " |")
    return "\n".join(lines)


def measure_latency_ms(pipe: Pipeline, sample: pd.DataFrame, n: int = 300) -> dict:
    """Tiempo de predict_proba para UNA fila, como en la API (medido en este equipo)."""
    pipe.predict_proba(sample)  # calentamiento
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        pipe.predict_proba(sample)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": round(float(np.percentile(times, 50)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3), "n": n}


def main() -> None:
    if not TRAINING_CSV.exists():
        raise SystemExit("Falta data/training.csv. Ejecuta: python ml/build_dataset.py")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAINING_CSV)
    X, y, groups = df[TELEMETRY_FIELDS], df[TARGET], df[GROUP]
    print(f"Dataset: {len(df)} filas, {groups.nunique()} flotas")
    print("Distribución de clases:", y.value_counts().reindex(CLASSES).to_dict())

    # 1. Split por flota
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    fleets_train = sorted(int(g) for g in groups.iloc[train_idx].unique())
    fleets_test = sorted(int(g) for g in groups.iloc[test_idx].unique())
    assert not set(fleets_train) & set(fleets_test), "Una flota quedó en entrenamiento y prueba"
    X_tr, y_tr, g_tr = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    print(f"Flotas de entrenamiento ({len(fleets_train)}): {fleets_train}")
    print(f"Flotas de prueba ({len(fleets_test)}): {fleets_test}")

    # 2. Validación cruzada por flota (solo entrenamiento) + 3. prueba
    cv = GroupKFold(n_splits=CV_FOLDS)
    candidates = [("baseline_mayoritaria", "completo")] + [
        (m, fs) for m in MODELS for fs in FEATURE_SETS]
    results = []
    for model_name, fs_name in candidates:
        feats = FEATURE_SETS[fs_name]
        t0 = time.perf_counter()
        scores = cross_val_score(build_pipeline(model_name, feats), X_tr, y_tr, groups=g_tr,
                                 cv=cv, scoring="f1_macro", n_jobs=1)
        pipe = build_pipeline(model_name, feats).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        r = {
            "modelo": model_name, "variables": fs_name,
            "cv_f1_macro_media": float(scores.mean()), "cv_f1_macro_desv": float(scores.std()),
            "prueba_f1_macro": float(f1_score(y_te, pred, average="macro")),
            "prueba_accuracy": float(accuracy_score(y_te, pred)),
            "segundos": round(time.perf_counter() - t0, 1),
        }
        results.append(r)
        print(f"{model_name:<24} {fs_name:<22} CV F1={r['cv_f1_macro_media']:.3f}±"
              f"{r['cv_f1_macro_desv']:.3f}  prueba F1={r['prueba_f1_macro']:.3f}")

    # Regla de selección (decidida antes de ver resultados):
    # a) mejor F1 macro medio en CV; b) entre los que quedan a menos de una desviación
    #    estándar del mejor, se prefiere el modelo más simple y luego el de menos variables.
    models_only = [r for r in results if r["modelo"] != "baseline_mayoritaria"]
    best = max(models_only, key=lambda r: r["cv_f1_macro_media"])
    threshold = best["cv_f1_macro_media"] - best["cv_f1_macro_desv"]
    tied = [r for r in models_only if r["cv_f1_macro_media"] >= threshold]
    chosen = min(tied, key=lambda r: (MODELS[r["modelo"]][0], len(FEATURE_SETS[r["variables"]]),
                                      -r["cv_f1_macro_media"]))
    features = FEATURE_SETS[chosen["variables"]]
    print(f"\nMejor en CV: {best['modelo']} / {best['variables']} ({best['cv_f1_macro_media']:.3f})")
    print(f"Elegido: {chosen['modelo']} / {chosen['variables']} (umbral de empate {threshold:.3f})")

    # 3. Matriz de confusión del elegido en las flotas de prueba
    pipe = build_pipeline(chosen["modelo"], features).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    cm = confusion_matrix(y_te, pred, labels=CLASSES)
    report_txt = classification_report(y_te, pred, labels=CLASSES, digits=3)
    print("\n" + report_txt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=axes[0], colorbar=False, values_format="d")
    axes[0].set_title("Conteos")
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred, labels=CLASSES, normalize="true"),
                           display_labels=CLASSES).plot(ax=axes[1], colorbar=False, values_format=".2f")
    axes[1].set_title("Proporción por clase real (recall)")
    fig.suptitle(f"{chosen['modelo']} · flotas de prueba · F1 macro {chosen['prueba_f1_macro']:.3f}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix_test.png", dpi=120)
    plt.close(fig)

    # 4. Modelo final: reentrenado con todas las flotas etiquetadas
    final = build_pipeline(chosen["modelo"], features).fit(X, y)
    clf = final.named_steps["clf"]
    if "n_jobs" in clf.get_params():
        clf.set_params(n_jobs=1)  # una fila por solicitud: los hilos solo agregan latencia
    joblib.dump(final, MODEL_PATH, compress=3)
    latency = measure_latency_ms(final, X.iloc[[0]])
    baseline = next(r for r in results if r["modelo"] == "baseline_mayoritaria")
    majority = y.value_counts().idxmax()

    meta = {
        "model_version": MODEL_VERSION,
        "model_name": chosen["modelo"],
        "feature_set": chosen["variables"],
        "features": features,
        "input_fields": TELEMETRY_FIELDS,
        "classes": [str(c) for c in clf.classes_],
        "majority_class_training": str(majority),
        "selection_rule": "max F1 macro medio en GroupKFold(5) por fleet_id; empate (< 1 desv.) -> modelo más simple y menos variables",
        "cv_f1_macro_mean": round(chosen["cv_f1_macro_media"], 4),
        "cv_f1_macro_std": round(chosen["cv_f1_macro_desv"], 4),
        "test_f1_macro": round(chosen["prueba_f1_macro"], 4),
        "test_f1_macro_baseline": round(baseline["prueba_f1_macro"], 4),
        "fleets_train": fleets_train,
        "fleets_test": fleets_test,
        "final_model_trained_on": "todas las flotas de data/training.csv (entrenamiento + prueba)",
        "rows": int(len(df)),
        "random_state": RANDOM_STATE,
        "single_row_latency_local": latency,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS_DIR / "train_metrics.json").write_text(
        json.dumps({"results": results, "chosen": chosen, "confusion_matrix_test": cm.tolist(),
                    "labels": CLASSES}, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Comparación de modelos", "",
        f"Generado por `ml/train_model.py` el {meta['trained_at_utc']} "
        f"(Python {meta['python']}, scikit-learn {meta['sklearn']}).", "",
        f"- Datos: {len(df)} filas de {groups.nunique()} flotas (`data/training.csv`).",
        f"- Separación por flota con `GroupShuffleSplit(test_size={TEST_SIZE}, random_state={RANDOM_STATE})`: "
        f"{len(fleets_train)} flotas de entrenamiento y {len(fleets_test)} de prueba, sin flotas compartidas.",
        f"- Selección con `GroupKFold({CV_FOLDS})` sobre las flotas de entrenamiento; métrica F1 macro.",
        "- Los tres clasificadores usan `class_weight=\"balanced\"` porque CRITICAL es cerca del 74 % de los datos.",
        "", "## Resultados", "",
        md_table([{k: v for k, v in r.items()} for r in results]), "",
        "## Modelo elegido", "",
        f"**{chosen['modelo']}** con variables **{chosen['variables']}**: {', '.join(features)}.", "",
        f"- F1 macro en validación cruzada: {chosen['cv_f1_macro_media']:.3f} ± {chosen['cv_f1_macro_desv']:.3f}.",
        f"- F1 macro en las flotas de prueba: {chosen['prueba_f1_macro']:.3f} "
        f"(línea base de clase mayoritaria: {baseline['prueba_f1_macro']:.3f}).",
        f"- Mejor resultado bruto en validación cruzada: {best['modelo']} / {best['variables']} "
        f"({best['cv_f1_macro_media']:.3f}). Umbral de empate: {threshold:.3f}.",
        f"- Latencia de `predict_proba` para una fila en el equipo de entrenamiento: "
        f"P50 {latency['p50_ms']} ms, P95 {latency['p95_ms']} ms ({latency['n']} repeticiones). "
        "No incluye red ni servidor.", "",
        "### Matriz de confusión (flotas de prueba)", "",
        "![matriz de confusión](confusion_matrix_test.png)", "",
        "```", report_txt.rstrip(), "```", "",
    ]
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nModelo guardado en {MODEL_PATH.relative_to(ROOT)}; reporte en reports/model_comparison.md")


if __name__ == "__main__":
    main()
