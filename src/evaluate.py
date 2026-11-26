"""Metrik evaluasi bersama untuk SEMUA model (ML klasik Tahap 4 dan CNN berikutnya).

Konvensi: kelas positif = 1 = COVID-19, negatif = 0 = healthy. Semua fungsi menerima
`y_true` (0/1) dan `y_score` (probabilitas kelas positif, atau skor monoton apa pun).

Isi:
  - youden_threshold()      : ambang keputusan yang memaksimalkan J = sensitivitas + spesifisitas − 1
  - threshold_metrics()     : metrik pada ambang tertentu (accuracy, balanced acc, precision, recall,
                              specificity, F1, confusion matrix)
  - bootstrap_ci()          : CI 95% persentil untuk ROC-AUC / PR-AUC (resampling rekaman, 1000x)
  - compute_metrics()       : gabungan semua metrik di atas → dict datar (siap disimpan)
  - curves()                : titik kurva ROC & PR
  - save_results()          : tambah/perbarui baris di reports/tables/hasil_semua_model.csv
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import load_config

RESULTS_FILE = "hasil_semua_model.csv"


# ---------------------------------------------------------------------------
# Ambang keputusan
# ---------------------------------------------------------------------------
def youden_threshold(y_true, y_score) -> float:
    """Ambang yang memaksimalkan Youden's J = TPR − FPR (dipilih di VAL, bukan test).

    roc_curve memberi ambang `t` dengan aturan prediksi positif bila skor ≥ t. Ambang pertama
    (inf) diabaikan. Bila seri, ambil ambang terbesar (lebih konservatif).
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    j[~np.isfinite(thr)] = -np.inf
    return float(thr[int(np.argmax(j))])


# ---------------------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------------------
def threshold_metrics(y_true, y_score, threshold: float) -> dict:
    """Metrik berbasis keputusan biner (skor ≥ threshold → positif)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": spec,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y_true, y_score, metric=roc_auc_score, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int | None = None) -> tuple[float, float]:
    """CI persentil (1−alpha) dengan bootstrap rekaman pada set uji.

    Sampel bootstrap yang hanya berisi satu kelas dilewati (AUC tidak terdefinisi).
    """
    seed = load_config().seed if seed is None else seed
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        vals.append(metric(yt, y_score[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_auc_diff(y_true, score_a, score_b, n_boot: int = 2000, seed: int | None = None) -> dict:
    """Selisih ROC-AUC (A − B) dua model pada set uji yang SAMA, dengan bootstrap berpasangan.

    Rekaman yang sama di-resample untuk kedua model → korelasi antar-model ikut diperhitungkan.
    Return: diff, CI 95% persentil, dan p dua sisi (proporsi bootstrap dengan tanda berlawanan × 2).
    """
    seed = load_config().seed if seed is None else seed
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    diff = roc_auc_score(y, a) - roc_auc_score(y, b)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d = np.asarray(d)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(diff), "ci_low": float(lo), "ci_high": float(hi), "p": float(p)}


def compute_metrics(y_true, y_score, threshold: float | None = None, n_boot: int = 1000,
                    seed: int | None = None) -> dict:
    """Semua metrik untuk satu model pada satu set data.

    threshold=None → pakai 0,5 (sebaiknya selalu berikan ambang yang dipilih di VAL).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    thr = 0.5 if threshold is None else threshold
    m = {"n": int(len(y_true)), "n_pos": int(y_true.sum()), "prevalensi": float(y_true.mean())}
    m["roc_auc"] = roc_auc_score(y_true, y_score)
    m["roc_auc_ci_low"], m["roc_auc_ci_high"] = bootstrap_ci(y_true, y_score, roc_auc_score, n_boot, seed=seed)
    m["pr_auc"] = average_precision_score(y_true, y_score)
    m["pr_auc_ci_low"], m["pr_auc_ci_high"] = bootstrap_ci(y_true, y_score, average_precision_score,
                                                           n_boot, seed=seed)
    m.update(threshold_metrics(y_true, y_score, thr))
    return m


def curves(y_true, y_score) -> dict:
    """Titik kurva ROC (fpr, tpr) dan PR (recall, precision)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}


def confusion(y_true, y_score, threshold: float) -> np.ndarray:
    """Confusion matrix 2×2 [[TN, FP], [FN, TP]] pada ambang tertentu."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return confusion_matrix(np.asarray(y_true).astype(int), y_pred, labels=[0, 1])


# ---------------------------------------------------------------------------
# Penyimpanan hasil
# ---------------------------------------------------------------------------
def save_results(model_name: str, experiment: str, metrics: dict, split: str = "test",
                 out=None) -> pd.DataFrame:
    """Tambah satu baris ke reports/tables/hasil_semua_model.csv (atau `out` bila diberikan).

    Baris dengan (model, eksperimen, split) yang sama diganti (bukan diduplikasi) agar script
    bisa dijalankan ulang. Kolom baru otomatis ditambahkan.
    """
    from pathlib import Path

    out = Path(out) if out is not None else load_config().paths.tables / RESULTS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {"model": model_name, "eksperimen": experiment, "split": split,
           **{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in metrics.items()},
           "waktu": datetime.now().strftime("%Y-%m-%d %H:%M")}
    new = pd.DataFrame([row])
    if out.exists():
        old = pd.read_csv(out)
        keep = ~((old["model"] == model_name) & (old["eksperimen"] == experiment) & (old["split"] == split))
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(out, index=False, encoding="utf-8")
    return new


def load_results(experiment: str | None = None) -> pd.DataFrame:
    path = load_config().paths.tables / RESULTS_FILE
    df = pd.read_csv(path)
    return df if experiment is None else df[df["eksperimen"] == experiment].reset_index(drop=True)
