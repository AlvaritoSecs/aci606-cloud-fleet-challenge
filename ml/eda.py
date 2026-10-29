"""Análisis exploratorio (EDA) del dataset multiflota.

Genera gráficos en reports/eda/ y un resumen en reports/eda.md.
Uso:
  python ml/build_dataset.py   # si data/training.csv no existe
  python ml/eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSES, FEATURES_ALL, REPORTS_DIR, ROOT, TARGET, TRAINING_CSV  # noqa: E402

COLORS = {"NORMAL": "#2e7d32", "WARNING": "#f9a825", "CRITICAL": "#c62828"}
EDA_DIR = REPORTS_DIR / "eda"


def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.2f}") -> str:
    frame = df.reset_index() if index else df
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    if not TRAINING_CSV.exists():
        raise SystemExit("Falta data/training.csv. Ejecuta: python ml/build_dataset.py")
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAINING_CSV)
    out: list[str] = ["# EDA del dataset multiflota", ""]
    out.append(f"Datos: `{TRAINING_CSV.relative_to(ROOT)}`, {len(df)} filas, "
               f"{df['fleet_id'].nunique()} flotas, {len(FEATURES_ALL)} variables candidatas. "
               f"Valores faltantes: {int(df[FEATURES_ALL].isna().sum().sum())}.")
    out.append("")

    # 1. Distribución de clases
    counts = df[TARGET].value_counts().reindex(CLASSES)
    share = (counts / len(df) * 100).round(1)
    dist = pd.DataFrame({"filas": counts, "porcentaje": share})
    out += ["## 1. Distribución de clases", "", md_table(dist.rename_axis("clase"), floatfmt="{:.1f}"), ""]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(CLASSES, counts.values, color=[COLORS[c] for c in CLASSES])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{share.iloc[i]} %", ha="center", va="bottom")
    ax.set_title("Distribución de clases (30 flotas)")
    ax.set_ylabel("filas")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)

    per_fleet = (pd.crosstab(df["fleet_id"], df[TARGET], normalize="index") * 100).reindex(columns=CLASSES)
    out.append(f"Por flota, el porcentaje de CRITICAL va de {per_fleet['CRITICAL'].min():.1f} % "
               f"a {per_fleet['CRITICAL'].max():.1f} % (mediana {per_fleet['CRITICAL'].median():.1f} %). "
               "CRITICAL es la clase mayoritaria: la línea base 'predecir siempre CRITICAL' tiene "
               "accuracy alta pero F1 macro bajo, por eso la métrica principal es F1 macro.")
    out.append("")

    # 2. Clases según la edad de la flota
    df["tramo"] = pd.cut(df["step"] / df["fleet_steps"], [0, .25, .5, .75, 1.0001],
                         right=False, labels=["0-25 %", "25-50 %", "50-75 %", "75-100 %"])
    by_time = (pd.crosstab(df["tramo"], df[TARGET], normalize="index") * 100).reindex(columns=CLASSES)
    out += ["## 2. Clases según el avance de la simulación", "",
            "Porcentaje de cada clase según el tramo de la vida de la flota:", "",
            md_table(by_time, floatfmt="{:.1f}"), "",
            "Los robots se degradan con el tiempo (la salud oculta tiene caídas bruscas), así que "
            "una flota recién iniciada tiene muchas más muestras NORMAL que una flota antigua. "
            "La flota de evaluación puede tener otra duración: el modelo no debe depender del tiempo.", ""]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bottom = None
    for c in CLASSES:
        ax.bar(by_time.index.astype(str), by_time[c], bottom=bottom, color=COLORS[c], label=c)
        bottom = by_time[c] if bottom is None else bottom + by_time[c]
    ax.set_ylabel("% de filas")
    ax.set_xlabel("avance de la simulación")
    ax.set_title("Las flotas pasan a CRITICAL con el tiempo")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "class_by_time.png", dpi=120)
    plt.close(fig)

    # 3. Variables por clase
    means = df.groupby(TARGET)[FEATURES_ALL].mean().reindex(CLASSES)
    out += ["## 3. Media de cada variable por clase", "", md_table(means.T.rename_axis("variable"), floatfmt="{:.3f}"), ""]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, feat in zip(axes.ravel(), FEATURES_ALL):
        ax.boxplot([df.loc[df[TARGET] == c, feat] for c in CLASSES], tick_labels=CLASSES, showfliers=False)
        ax.set_title(feat)
    fig.suptitle("Distribución de cada variable por clase (sin valores atípicos)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "features_by_class.png", dpi=120)
    plt.close(fig)

    # 4. Correlaciones con la clase ordinal y con el tiempo
    ordinal = df[TARGET].map({"NORMAL": 0, "WARNING": 1, "CRITICAL": 2})
    corr = pd.DataFrame({
        "corr. con gravedad (0=NORMAL, 2=CRITICAL)": df[FEATURES_ALL].corrwith(ordinal),
        "corr. con el paso de tiempo": df[FEATURES_ALL].corrwith(df["step"]),
    })
    out += ["## 4. Correlaciones", "", md_table(corr.rename_axis("variable"), floatfmt="{:.3f}"), "",
            "Lectura: `motor_temp`, `vibration` y `motor_current` son las señales directas del deterioro. "
            "`battery` se correlaciona con la gravedad, pero sobre todo con el tiempo: baja por el uso y "
            "la flota se deteriora en paralelo, sin que la batería cause el estado. Esa relación puede no "
            "repetirse en una flota de otra duración, así que se evalúa el modelo con y sin `battery`. "
            "`obstacle_distance` casi no se relaciona con la clase.", ""]
    fig, ax = plt.subplots(figsize=(7, 5))
    mat = df[FEATURES_ALL].assign(gravedad=ordinal, paso=df["step"]).corr()
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(mat)), mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat)), mat.columns)
    for i in range(len(mat)):
        for j in range(len(mat)):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "correlation.png", dpi=120)
    plt.close(fig)

    out += ["## Gráficos", "",
            "- `eda/class_distribution.png`", "- `eda/class_by_time.png`",
            "- `eda/features_by_class.png`", "- `eda/correlation.png`", ""]
    (REPORTS_DIR / "eda.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
