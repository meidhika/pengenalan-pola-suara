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


# ---------------------------------------------------------------------------
# Uji DeLong (Tahap 6)
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    """Peringkat tengah (rata-rata untuk nilai seri), 1-based."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    r = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        r[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = r
    return out


def delong_test(y_true, score_a, score_b) -> dict:
    """Uji DeLong (DeLong dkk., 1988; algoritma cepat Sun & Xu, 2014) untuk dua AUC berkorelasi.

    Kedua skor dihitung pada rekaman yang SAMA. H0: AUC_A = AUC_B. Return: auc_a, auc_b, diff,
    se, z, p (dua sisi), CI 95% selisih (pendekatan normal).
    """
    from scipy.stats import norm

    y = np.asarray(y_true).astype(int)
    S = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    pos, neg = S[:, y == 1], S[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = S.shape[0]
    tx = np.array([_midrank(pos[i]) for i in range(k)])
    ty = np.array([_midrank(neg[i]) for i in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[i], neg[i]])) for i in range(k)])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n          # komponen struktural positif
    v10 = 1 - (tz[:, m:] - ty) / m      # komponen struktural negatif
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    se = float(np.sqrt(max(var, 1e-12)))
    z = diff / se
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "se": se, "z": float(z),
            "p": float(2 * norm.sf(abs(z))), "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se}


def subgroup_metrics(y_true, y_score, groups, threshold: float, n_boot: int = 500, min_pos: int = 5,
                     seed: int | None = None) -> pd.DataFrame:
    """Metrik per subgrup (fairness): n, n_pos, AUC [CI 95%], sensitivitas, spesifisitas, precision.

    AUC hanya dihitung bila subgrup punya ≥ `min_pos` positif dan ≥ `min_pos` negatif; selain itu NaN.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, float)
    g = pd.Series(np.asarray(groups)).astype(str).values
    rows = []
    for name in pd.unique(g):
        mk = g == name
        yy, ss = y[mk], s[mk]
        r = {"subgrup": name, "n": int(mk.sum()), "n_pos": int(yy.sum()),
             "prevalensi": float(yy.mean()) if mk.any() else np.nan}
        if yy.sum() >= min_pos and (1 - yy).sum() >= min_pos:
            r["roc_auc"] = roc_auc_score(yy, ss)
            r["roc_auc_ci_low"], r["roc_auc_ci_high"] = bootstrap_ci(yy, ss, roc_auc_score, n_boot, seed=seed)
        else:
            r["roc_auc"] = r["roc_auc_ci_low"] = r["roc_auc_ci_high"] = np.nan
        tm = threshold_metrics(yy, ss, threshold)
        r.update({k: tm[k] for k in ["recall_sensitivity", "specificity", "precision"]})
        if yy.sum() == 0:
            r["recall_sensitivity"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)
