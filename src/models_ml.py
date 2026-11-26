"""Model ML klasik (Tahap 4) untuk fitur tabular: pipeline sklearn/imblearn + ruang pencarian hyperparameter.

Pipeline (semua langkah di-fit HANYA pada data latih fold/train → bebas leakage):
    StandardScaler → SelectKBest(f_classif) [k ∈ grid, 'all' = tanpa seleksi] → [SMOTE] → classifier

Penanganan ketidakseimbangan kelas (dibandingkan):
    - "balanced": class_weight='balanced' (LogReg, SVM, RF) / scale_pos_weight = n_neg/n_pos (XGBoost).
                  k-NN tidak punya bobot kelas → varian ini = tanpa penanganan ("none").
    - "smote"   : SMOTE di dalam pipeline imblearn → oversampling sintetis hanya pada fold latih;
                  fold validasi/val/test tidak pernah disentuh.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import load_config

# Nama tampilan model
NAMA_MODEL = {
    "dummy": "Dummy (stratified)",
    "logreg": "Logistic Regression",
    "svm": "SVM (RBF)",
    "knn": "k-NN",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}
ALGORITMA = list(NAMA_MODEL)


def imbalance_variants(algo: str) -> list[str]:
    """Varian penanganan imbalance yang dibandingkan untuk tiap algoritma."""
    if algo == "dummy":
        return ["none"]
    if algo == "knn":
        return ["none", "smote"]
    return ["balanced", "smote"]


def make_classifier(algo: str, imbalance: str, y_train=None, seed: int | None = None):
    seed = load_config().seed if seed is None else seed
    cw = "balanced" if imbalance == "balanced" else None
    if algo == "dummy":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if algo == "logreg":
        return LogisticRegression(class_weight=cw, max_iter=5000, random_state=seed)
    if algo == "svm":
        return SVC(kernel="rbf", probability=True, class_weight=cw, random_state=seed)
    if algo == "knn":
        return KNeighborsClassifier()
    if algo == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=1, random_state=seed)
    if algo == "xgb":
        spw = 1.0
        if imbalance == "balanced" and y_train is not None:
            y = np.asarray(y_train)
            spw = float((y == 0).sum() / max((y == 1).sum(), 1))
        return XGBClassifier(n_estimators=300, tree_method="hist", eval_metric="logloss",
                             scale_pos_weight=spw, n_jobs=1, random_state=seed, verbosity=0)
    raise ValueError(f"algoritma tidak dikenal: {algo}")


def make_pipeline(algo: str, imbalance: str, y_train=None, seed: int | None = None) -> Pipeline:
    """Scaler → SelectKBest → [SMOTE] → classifier (imblearn Pipeline)."""
    seed = load_config().seed if seed is None else seed
    steps = [("scaler", StandardScaler()), ("select", SelectKBest(f_classif, k="all"))]
    if imbalance == "smote":
        steps.append(("smote", SMOTE(k_neighbors=5, random_state=seed)))
    steps.append(("clf", make_classifier(algo, imbalance, y_train, seed)))
    return Pipeline(steps)


def k_grid(n_features: int) -> list:
    """Pilihan k untuk SelectKBest yang valid untuk jumlah fitur tertentu."""
    ks = [k for k in (20, 50, 100) if k < n_features]
    return ks + ["all"]


def param_space(algo: str, n_features: int) -> dict:
    """Ruang pencarian RandomizedSearchCV (distribusi kecil agar cepat)."""
    sp = {"select__k": k_grid(n_features)}
    if algo == "dummy":
        return {}
    if algo == "logreg":
        sp.update({"clf__C": loguniform(1e-4, 1e1)})
    elif algo == "svm":
        sp.update({"clf__C": loguniform(1e-2, 1e2), "clf__gamma": loguniform(1e-4, 1e-1)})
    elif algo == "knn":
        sp.update({"clf__n_neighbors": randint(5, 151), "clf__weights": ["uniform", "distance"],
                   "clf__p": [1, 2]})
    elif algo == "rf":
        sp.update({"clf__max_depth": [None, 5, 10, 20], "clf__min_samples_leaf": [1, 2, 5, 10, 20],
                   "clf__max_features": ["sqrt", 0.1, 0.3]})
    elif algo == "xgb":
        sp.update({"clf__max_depth": [2, 3, 4, 6], "clf__learning_rate": loguniform(1e-2, 3e-1),
                   "clf__subsample": [0.6, 0.8, 1.0], "clf__colsample_bytree": [0.3, 0.5, 0.8],
                   "clf__min_child_weight": [1, 5, 10], "clf__n_estimators": [100, 300, 500]})
    return sp


# ---------------------------------------------------------------------------
# Kelompok fitur untuk ablasi
# ---------------------------------------------------------------------------
_SPEKTRAL = ("centroid_", "bandwidth_", "rolloff_", "flatness_", "contrast_")


def feature_subsets(cols: list[str]) -> dict[str, list[str]]:
    """Subset fitur untuk ablasi. MFCC = koefisien statis + delta + delta-delta.

    'waktu' = RMS & ZCR saja (durasi dipisah karena berpotensi menjadi *shortcut*).
    """
    waktu = [c for c in cols if c.startswith(("rms_", "zcr_"))]
    spek = [c for c in cols if c.startswith(_SPEKTRAL)]
    mfcc = [c for c in cols if c.startswith("mfcc_")]
    gfcc = [c for c in cols if c.startswith("gfcc_")]
    return {
        "(i) waktu (RMS, ZCR)": waktu,
        "(ii) spektral": spek,
        "(iii) MFCC (+Δ, ΔΔ)": mfcc,
        "(iv) GFCC": gfcc,
        "(v) MFCC + GFCC": mfcc + gfcc,
        "(vi) semua": list(cols),
        "(vii) durasi saja": ["durasi_s"],
        "(viii) semua tanpa durasi": [c for c in cols if c != "durasi_s"],
    }


def selected_feature_names(pipe: Pipeline, cols: list[str]) -> list[str]:
    """Nama fitur yang lolos SelectKBest dalam pipeline terlatih."""
    mask = pipe.named_steps["select"].get_support()
    return [c for c, m in zip(cols, mask) if m]


def builtin_importance(pipe: Pipeline, cols: list[str]) -> dict[str, float]:
    """feature_importances_ (RF: mean decrease impurity; XGBoost: gain ternormalisasi)."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return {}
    return dict(zip(selected_feature_names(pipe, cols), clf.feature_importances_))
