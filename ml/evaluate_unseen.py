"""Evaluación final del modelo guardado en flotas NUNCA usadas (data/unseen.csv).

Imita la evaluación del profesor: otras semillas, otros largos y otra cantidad de robots.
Ninguna decisión de modelado se tomó mirando estas flotas.

Uso:
  python ml/evaluate_unseen.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSES, META_PATH, MODEL_PATH, REPORTS_DIR, TARGET, TELEMETRY_FIELDS, UNSEEN_CSV  # noqa: E402


def f1(y_true, y_pred) -> float:
    # labels=CLASSES: una clase ausente en una flota cuenta como F1 = 0, como en la evaluación global
    return float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))


def main() -> None:
    for path, hint in ((UNSEEN_CSV, "python ml/build_dataset.py"), (MODEL_PATH, "python ml/train_model.py")):
        if not path.exists():
            raise SystemExit(f"Falta {path.name}. Ejecuta: {hint}")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(UNSEEN_CSV)
    df["pred"] = model.predict(df[TELEMETRY_FIELDS])
    df["base"] = meta["majority_class_training"]

    total_model, total_base = f1(df[TARGET], df["pred"]), f1(df[TARGET], df["base"])
    report_txt = classification_report(df[TARGET], df["pred"], labels=CLASSES, digits=3)

    rows = []
    for fleet, g in df.groupby("fleet_id"):
        rows.append({
            "flota": int(fleet), "robots": int(g["robot_id"].nunique()), "pasos": int(g["fleet_steps"].iloc[0]),
            "filas": len(g), "% CRITICAL": round(100 * (g[TARGET] == "CRITICAL").mean(), 1),
            "F1 modelo": round(f1(g[TARGET], g["pred"]), 3), "F1 línea base": round(f1(g[TARGET], g["base"]), 3),
        })
    fleets = pd.DataFrame(rows)

    df["tramo"] = pd.cut(df["step"] / df["fleet_steps"], [0, .25, .5, .75, 1.0001], right=False,
                         labels=["0-25 %", "25-50 %", "50-75 %", "75-100 %"])
    by_time = [{"tramo": str(t), "filas": len(g), "F1 modelo": round(f1(g[TARGET], g["pred"]), 3)}
               for t, g in df.groupby("tramo", observed=True)]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ConfusionMatrixDisplay(confusion_matrix(df[TARGET], df["pred"], labels=CLASSES, normalize="true"),
                           display_labels=CLASSES).plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(f"Flotas no vistas · F1 macro {total_model:.3f}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix_unseen.png", dpi=120)
    plt.close(fig)

    def table(records: list[dict]) -> str:
        cols = list(records[0])
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in records]
        return "\n".join(out)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = [
        "# Evaluación en flotas no vistas", "",
        f"Generado por `ml/evaluate_unseen.py` el {now} con el modelo `{meta['model_name']}` "
        f"({meta['model_version']}, variables: {', '.join(meta['features'])}).", "",
        f"{len(df)} filas de {df['fleet_id'].nunique()} flotas con semillas reservadas, que no se usaron "
        "para entrenar ni para elegir el modelo.", "",
        f"- **F1 macro del modelo: {total_model:.3f}**",
        f"- F1 macro de la línea base (siempre {meta['majority_class_training']}): {total_base:.3f}", "",
        "## Por flota", "", table(fleets.to_dict("records")), "",
        "## Según el avance de la simulación", "", table(by_time), "",
        "## Detalle por clase", "", "```", report_txt.rstrip(), "```", "",
        "![matriz de confusión](confusion_matrix_unseen.png)", "",
    ]
    (REPORTS_DIR / "unseen_evaluation.md").write_text("\n".join(md), encoding="utf-8")
    (REPORTS_DIR / "unseen_metrics.json").write_text(json.dumps({
        "f1_macro_model": round(total_model, 4), "f1_macro_baseline": round(total_base, 4),
        "per_fleet": rows, "by_time": by_time, "evaluated_at_utc": now}, indent=2), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
