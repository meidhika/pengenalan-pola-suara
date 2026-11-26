"""Analisis akhir (Tahap 6): konsolidasi prediksi, uji silang label, analisis error & subgrup.

Semua fungsi memakai model TERSIMPAN di models/ (tidak ada training ulang).

Isi:
  - NAMA_MODEL, load_scores()       : skor val/test semua model (ML + CNN) dalam satu tabel per eksperimen
  - thresholds_from_results()       : ambang Youden (dipilih di val) dari hasil_semua_model.csv
  - final_table()                   : tabel final metrik + penanda model terbaik
  - to_markdown()                   : DataFrame → tabel Markdown (tanpa dependensi tabulate)
  - ml_predict(), cnn_predict()     : inferensi model tersimpan pada data lain (uji silang A↔B)
  - seen_uuids()                    : uuid yang dipakai saat training/seleksi (dikecualikan dari uji silang)
  - add_subgroups()                 : kolom subgrup demografi (gender, kelompok umur, kondisi, demam)
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.config import load_config

# kolom prediksi → nama tampilan (sama dengan hasil_semua_model.csv)
NAMA_MODEL = {
    "dummy": "Dummy (stratified)", "logreg": "Logistic Regression", "svm": "SVM (RBF)", "knn": "k-NN",
    "rf": "Random Forest", "xgb": "XGBoost", "cnn_logmel_s42": "CNN (log-mel)", "cnn_mfcc2d_s42": "CNN (MFCC 2D)",
}
URUTAN_MODEL = list(NAMA_MODEL.values())


def load_scores(experiment: str = "A") -> pd.DataFrame:
    """Gabungkan skor val/test ML (+ CNN seed 42 untuk A). Kolom: uuid, label, split, <nama model>..."""
    d = load_config().paths.models
    df = pd.read_parquet(d / f"preds_ml_{experiment}.parquet")
    cnn_path = d / f"preds_cnn_{experiment}.parquet"
    if cnn_path.exists():
        cnn = pd.read_parquet(cnn_path)
        df = df.merge(cnn.drop(columns=["label", "split"]), on="uuid", how="left", validate="1:1")
    keep = ["uuid", "label", "split"] + [c for c in NAMA_MODEL if c in df.columns]
    extra = [c for c in df.columns if c.startswith("cnn_") and c not in keep]
    return df[keep + extra].rename(columns=NAMA_MODEL)


def thresholds_from_results(experiment: str = "A") -> dict:
    from src.evaluate import load_results
    r = load_results(experiment)
    r = r[r["split"] == "test"]
    return dict(zip(r["model"], r["threshold"]))


def final_table(experiment: str = "A") -> pd.DataFrame:
    """Tabel final (split test) + penanda: ★ = AUC val tertinggi (kriteria seleksi yang ditetapkan
    sebelum melihat test), ▲ = AUC test tertinggi (informasi saja; bukan kriteria seleksi)."""
    from src.evaluate import load_results
    r = load_results(experiment)
    r = r[r["split"] == "test"].copy()
    r["urut"] = r["model"].map({m: i for i, m in enumerate(URUTAN_MODEL)})
    r = r.sort_values("urut").drop(columns="urut").reset_index(drop=True)
    non_dummy = r["model"] != "Dummy (stratified)"
    r["terbaik_val"] = False
    r["terbaik_test"] = False
    r.loc[r[non_dummy]["val_auc"].idxmax(), "terbaik_val"] = True
    r.loc[r[non_dummy]["roc_auc"].idxmax(), "terbaik_test"] = True
    return r


def fmt_final_table(r: pd.DataFrame) -> pd.DataFrame:
    """Format tampilan (3 desimal, AUC [CI]) untuk CSV/Markdown laporan."""
    def f(x):
        return "–" if pd.isna(x) else f"{x:.3f}"

    out = pd.DataFrame({
        "Model": r["model"] + np.where(r["terbaik_val"], " ★", "") + np.where(r["terbaik_test"], " ▲", ""),
        "Accuracy": r["accuracy"].map(f), "Balanced Acc": r["balanced_accuracy"].map(f),
        "Precision": r["precision"].map(f), "Recall (Sens.)": r["recall_sensitivity"].map(f),
        "Specificity": r["specificity"].map(f), "F1": r["f1"].map(f),
        "ROC-AUC [95% CI]": [f"{a:.3f} [{lo:.3f}–{hi:.3f}]" for a, lo, hi in
                             zip(r["roc_auc"], r["roc_auc_ci_low"], r["roc_auc_ci_high"])],
        "PR-AUC": r["pr_auc"].map(f), "AUC val": r["val_auc"].map(f), "Ambang": r["threshold"].map(f),
    })
    if "cnn_seed_auc_mean" in r:
        out["AUC test 3 seed (mean ± std)"] = [
            "–" if pd.isna(m) else f"{m:.3f} ± {s:.3f}" for m, s in zip(r["cnn_seed_auc_mean"], r["cnn_seed_auc_std"])]
    return out


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(x):
        if isinstance(x, (float, np.floating)):
            return "–" if np.isnan(x) else floatfmt.format(x)
        return str(x)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row.values) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Inferensi model tersimpan pada data lain (uji silang)
# ---------------------------------------------------------------------------
def ml_predict(algo: str, trained_on: str, tab: pd.DataFrame) -> np.ndarray:
    """Skor P(COVID-19) dari models/ml_<algo>_<trained_on>.joblib untuk baris `tab` (fitur tabular)."""
    m = joblib.load(load_config().paths.models / f"ml_{algo}_{trained_on}.joblib")
    return m["pipeline"].predict_proba(tab[m["features"]])[:, 1]


def cnn_predict(ckpt_name: str, data_experiment: str, uuids: list[str] | None = None) -> pd.DataFrame:
    """Skor CNN tersimpan pada fitur 2D eksperimen lain (train/val/test digabung).

    Normalisasi memakai statistik train dari checkpoint (bukan di-fit ulang pada data baru).
    Return DataFrame [uuid, skor].
    """
    import torch
    from torch.utils.data import DataLoader

    import src.models_cnn as mc

    ck = torch.load(load_config().paths.models / ckpt_name, map_location="cpu", weights_only=False)
    model = mc.build_model(ck["config"])
    model.load_state_dict(ck["model"])
    norm = mc.BandNormalizer()
    norm.mean_, norm.std_ = ck["normalizer"]["mean"], ck["normalizer"]["std"]
    rows = []
    for split in ["train", "val", "test"]:
        X, y, u, _ = mc.load_split(ck["feature"], data_experiment, split, ck["time_pool"])
        if uuids is not None:
            mk = np.isin(u, list(uuids))
            X, y, u = X[mk], y[mk], u[mk]
        if len(y) == 0:
            continue
        dl = DataLoader(mc.SpecDataset(norm.transform(X), y), batch_size=64, shuffle=False)
        rows.append(pd.DataFrame({"uuid": u, "skor": mc.predict_proba(model, dl, torch.device("cpu"))}))
    return pd.concat(rows, ignore_index=True)


def seen_uuids(experiment: str) -> set[str]:
    """uuid yang benar-benar DIPAKAI model sumber: manifest train (setelah undersampling) ∪ val
    (val dipakai untuk ambang & seleksi model). Rekaman healthy train yang dibuang undersampling tidak
    pernah dilihat model (scaler/SMOTE/normalizer di-fit pada manifest train) sehingga boleh dievaluasi."""
    s = pd.read_csv(load_config().paths.processed / f"manifest_{experiment}.csv")
    return set(s.loc[s["split"].isin(["train", "val"]), "uuid"])


# ---------------------------------------------------------------------------
# Subgrup demografi
# ---------------------------------------------------------------------------
KOLOM_SUBGRUP = {"gender": "Gender", "kelompok_umur": "Kelompok umur",
                 "respiratory_condition": "Kondisi pernapasan", "fever_muscle_pain": "Demam / nyeri otot"}


def add_subgroups(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan metadata (uuid) dan buat kolom subgrup berlabel Bahasa Indonesia."""
    m = meta[["uuid", "age", "gender", "respiratory_condition", "fever_muscle_pain"]].copy()
    bins = [0, 24, 34, 44, 54, 200]
    lab = ["< 25", "25–34", "35–44", "45–54", "≥ 55"]
    m["kelompok_umur"] = pd.cut(m["age"], bins=bins, labels=lab).astype(object).fillna("tidak diisi")
    m["gender"] = m["gender"].fillna("tidak diisi")
    for c in ["respiratory_condition", "fever_muscle_pain"]:
        m[c] = m[c].map({True: "ya", False: "tidak"}).fillna("tidak diisi")
    return df.merge(m.drop(columns="age"), on="uuid", how="left", validate="m:1")
