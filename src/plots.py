"""Utilitas grafik & tabel untuk laporan (PNG dpi 200, CSV)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import load_config

DPI = 200


def setup_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": DPI, "savefig.bbox": "tight"})


def save_fig(fig, name: str):
    """Simpan gambar ke reports/figures/<name>.png."""
    out = load_config().paths.figures / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    return out


def save_table(df: pd.DataFrame, name: str, index: bool = True):
    """Simpan tabel ke reports/tables/<name>.csv."""
    out = load_config().paths.tables / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index, encoding="utf-8")
    return out


def bar_counts(s: pd.Series, title: str, ax=None, dropna: bool = False, order=None):
    """Bar chart value_counts dengan label jumlah & persentase di atas batang."""
    vc = s.astype("object").fillna("(kosong)").value_counts() if not dropna else s.value_counts()
    if order is not None:
        vc = vc.reindex([o for o in order if o in vc.index])
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, color="#4C72B0")
    tot = vc.sum()
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v}\n({v / tot:.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Jumlah rekaman")
    ax.set_xlabel("")
    ax.margins(y=0.2)
    ax.tick_params(axis="x", rotation=20)
    return ax
