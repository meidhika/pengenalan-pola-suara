"""Tahap 4 — klasifikasi dengan ML klasik pada fitur tabular (Eksperimen A lalu B).

Protokol (anti-leakage; dipilih & didokumentasikan):
  1. Tuning hyperparameter: RandomizedSearchCV (n_iter ≤ 20, scoring ROC-AUC) dengan CV pada TRAIN saja.
       - A: StratifiedKFold 5-fold.
       - B (data kecil): RepeatedStratifiedKFold 5-fold × 3 ulangan.
     Varian imbalance ("balanced" vs "smote") diperlakukan sebagai bagian dari tuning: dipilih
     berdasarkan AUC CV di train. Seluruh langkah pipeline (scaler, SelectKBest, SMOTE) di-fit per fold.
  2. VAL dipakai untuk: (a) memilih ambang keputusan (Youden's J), (b) memilih model terbaik
     antar-algoritma (AUC val) untuk ablasi & interpretasi. Model TIDAK dilatih ulang dengan val.
  3. TEST dievaluasi SEKALI per model final dengan ambang dari val; CI 95% AUC via bootstrap 1000x.
  4. Estimasi variabilitas: repeated stratified CV 5×5 pada train (mean ± std AUC) dengan hyperparameter terbaik.
  5. Ablasi fitur dengan algoritma terbaik (tuning ulang per subset, n_iter lebih kecil).
  6. Importance: bawaan RF & XGBoost + permutation importance (ROC-AUC, 10 ulangan) di VAL.

Keluaran:
  models/ml_<algo>_<X>.joblib                    dict: pipeline, threshold, fitur, ringkasan tuning, prediksi
  models/preds_ml_<X>.parquet                    prediksi val/test semua model (untuk kurva ROC/PR)
  reports/tables/hasil_semua_model.csv           (via src.evaluate.save_results)
  reports/tables/04_imbalance_<X>.csv            balanced vs SMOTE per algoritma
  reports/tables/04_ablasi_fitur.csv             (A; B → 04_ablasi_fitur_B.csv)
  reports/tables/04_importance_<X>.csv
  reports/logs/train_ml_<X>.log

Contoh:
  python scripts/04_train_ml.py --limit 200 --n-iter 3       # uji cepat → reports/logs/uji_limit/
  python scripts/04_train_ml.py --experiments A B             # run penuh (model yang sudah ada dilewati)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src import models_ml as mm  # noqa: E402
from src.config import load_config, set_seed  # noqa: E402

META = ["uuid", "split", "label", "label_name"]
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)  # f_classif pada kolom konstan (subset kecil)


def setup_logger(path: Path) -> logging.Logger:
    log = logging.getLogger("ml")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def load_data(exp: str, cfg, limit: int | None, seed: int):
    df = pd.read_parquet(cfg.paths.features / f"tabular_{exp}.parquet")
    feats = [c for c in df.columns if c not in META]
    parts = {}
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        if limit:  # subsampel berstrata untuk uji cepat
            n = len(d)
            d = pd.concat([g.sample(min(len(g), max(10, limit * len(g) // n)), random_state=seed)
                           for _, g in d.groupby("label")])
        parts[s] = d.reset_index(drop=True)
    return parts, feats


def make_cv(exp: str, seed: int, repeats: int | None = None):
    if exp == "B":
        return RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats or 3, random_state=seed)
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def tune(algo, imb, X, y, cols, cv, n_iter, n_jobs, seed):
    """RandomizedSearchCV satu (algoritma, varian imbalance). Mengembalikan (estimator, ringkasan)."""
    pipe = mm.make_pipeline(algo, imb, y_train=y, seed=seed)
    space = mm.param_space(algo, len(cols))
    t0 = time.time()
    if not space:  # dummy: tanpa hyperparameter
        sc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs)
        est = clone(pipe).fit(X, y)
        return est, {"cv_auc_mean": sc.mean(), "cv_auc_std": sc.std(), "best_params": {},
                     "detik": time.time() - t0}
    # jumlah kombinasi diskret bisa < n_iter
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc", cv=cv, n_jobs=n_jobs,
                                random_state=seed, refit=True, error_score="raise")
    search.fit(X, y)
    i = search.best_index_
    return search.best_estimator_, {
        "cv_auc_mean": float(search.cv_results_["mean_test_score"][i]),
        "cv_auc_std": float(search.cv_results_["std_test_score"][i]),
        "best_params": {k: (v if isinstance(v, (str, int, type(None))) else float(v))
                        for k, v in search.best_params_.items()},
        "detik": time.time() - t0,
    }


def proba(est, X):
    return est.predict_proba(X)[:, 1]


def run_experiment(exp: str, args, cfg, log):
    seed = set_seed()
    limit_mode = args.limit is not None
    out_models = cfg.paths.models if not limit_mode else cfg.paths.logs / "uji_limit"
    out_tables = cfg.paths.tables if not limit_mode else cfg.paths.logs / "uji_limit"
    out_models.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    res_file = None if not limit_mode else out_tables / "hasil_semua_model.csv"

    parts, feats = load_data(exp, cfg, args.limit, seed)
    Xtr, ytr = parts["train"][feats], parts["train"]["label"].to_numpy()
    Xva, yva = parts["val"][feats], parts["val"]["label"].to_numpy()
    Xte, yte = parts["test"][feats], parts["test"]["label"].to_numpy()
    log.info(f"=== Eksperimen {exp}: {len(feats)} fitur | train {len(ytr)} (pos {ytr.sum()}) | "
             f"val {len(yva)} (pos {yva.sum()}) | test {len(yte)} (pos {yte.sum()})")
    cv = make_cv(exp, seed)
    cv_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)

    preds = {"val": pd.DataFrame({"uuid": parts["val"]["uuid"], "label": yva}),
             "test": pd.DataFrame({"uuid": parts["test"]["uuid"], "label": yte})}
    imb_rows, final = [], {}

    for algo in tqdm(mm.ALGORITMA, desc=f"model {exp}"):
        path = out_models / f"ml_{algo}_{exp}.joblib"
        if path.exists() and not args.overwrite:
            obj = joblib.load(path)
            log.info(f"[{algo}] sudah ada → dimuat ({path.name})")
        else:
            variants = {}
            for imb in mm.imbalance_variants(algo):
                est, summ = tune(algo, imb, Xtr, ytr, feats, cv, args.n_iter, args.n_jobs, seed)
                pva, pte = proba(est, Xva), proba(est, Xte)
                thr = ev.youden_threshold(yva, pva)
                summ.update({"val_auc": roc_auc_score(yva, pva), "threshold": thr})
                variants[imb] = (est, summ, pva, pte)
                log.info(f"[{algo}/{imb}] CV AUC {summ['cv_auc_mean']:.3f}±{summ['cv_auc_std']:.3f} | "
                         f"val AUC {summ['val_auc']:.3f} | {summ['detik']:.0f}s | {summ['best_params']}")
            # pilih varian imbalance berdasarkan AUC CV di train (bukan val/test)
            best_imb = max(variants, key=lambda k: variants[k][1]["cv_auc_mean"])
            est, summ, pva, pte = variants[best_imb]
            rep = cross_val_score(clone(est), Xtr, ytr, cv=cv_rep, scoring="roc_auc", n_jobs=args.n_jobs)
            summ.update({"cvrep_auc_mean": float(rep.mean()), "cvrep_auc_std": float(rep.std())})
            obj = {"pipeline": est, "algo": algo, "eksperimen": exp, "imbalance": best_imb,
                   "threshold": summ["threshold"], "features": feats, "summary": summ,
                   "variants": {k: {**v[1], "test_auc": roc_auc_score(yte, v[3]),
                                    "test_metrics": ev.threshold_metrics(yte, v[3], v[1]["threshold"])}
                                for k, v in variants.items()},
                   "pred_val": pva, "pred_test": pte,
                   "cv_info": f"{type(cv).__name__}; tuning di train; ambang Youden di val"}
            joblib.dump(obj, path)

        s = obj["summary"]
        m = ev.compute_metrics(yte, obj["pred_test"], obj["threshold"], n_boot=args.n_boot, seed=seed)
        m.update({"imbalance": obj["imbalance"], "cv_auc_mean": s["cv_auc_mean"], "cv_auc_std": s["cv_auc_std"],
                  "cvrep5x5_auc_mean": s["cvrep_auc_mean"], "cvrep5x5_auc_std": s["cvrep_auc_std"],
                  "val_auc": s["val_auc"], "best_params": json.dumps(s["best_params"])})
        ev.save_results(mm.NAMA_MODEL[algo], exp, m, out=res_file)
        final[algo] = obj
        preds["val"][algo] = obj["pred_val"]
        preds["test"][algo] = obj["pred_test"]
        for imb, v in obj["variants"].items():
            imb_rows.append({"model": mm.NAMA_MODEL[algo], "imbalance": imb, "dipilih": imb == obj["imbalance"],
                             "cv_auc_mean": v["cv_auc_mean"], "cv_auc_std": v["cv_auc_std"],
                             "val_auc": v["val_auc"], "test_auc": v["test_auc"],
                             "test_sensitivity": v["test_metrics"]["recall_sensitivity"],
                             "test_specificity": v["test_metrics"]["specificity"],
                             "test_f1": v["test_metrics"]["f1"], "threshold": v["threshold"],
                             "best_params": json.dumps(v["best_params"])})
        log.info(f"[{algo}] FINAL ({obj['imbalance']}) test AUC {m['roc_auc']:.3f} "
                 f"[{m['roc_auc_ci_low']:.3f}–{m['roc_auc_ci_high']:.3f}] | PR-AUC {m['pr_auc']:.3f} | "
                 f"sens {m['recall_sensitivity']:.3f} spes {m['specificity']:.3f} F1 {m['f1']:.3f}")

    pd.DataFrame(imb_rows).to_csv(out_tables / f"04_imbalance_{exp}.csv", index=False)
    pd.concat([d.assign(split=s) for s, d in preds.items()]).to_parquet(out_models / f"preds_ml_{exp}.parquet")

    # ---------------- model terbaik (AUC val, tanpa dummy) ----------------
    cand = {a: o for a, o in final.items() if a != "dummy"}
    best = max(cand, key=lambda a: cand[a]["summary"]["val_auc"])
    log.info(f"Model terbaik (AUC val): {best} ({cand[best]['imbalance']}), "
             f"val AUC {cand[best]['summary']['val_auc']:.3f}")
    info = {"eksperimen": exp, "model_terbaik": best, "imbalance": cand[best]["imbalance"],
            "val_auc": cand[best]["summary"]["val_auc"]}

    # ---------------- importance ----------------
    if not args.skip_importance:
        rows = []
        for algo in ("rf", "xgb"):
            for f, v in mm.builtin_importance(final[algo]["pipeline"], feats).items():
                rows.append({"metode": f"bawaan_{algo}", "fitur": f, "importance": v, "std": np.nan})
        pi = permutation_importance(final[best]["pipeline"], Xva, yva, scoring="roc_auc", n_repeats=10,
                                    random_state=seed, n_jobs=args.n_jobs)
        for f, mu, sd in zip(feats, pi.importances_mean, pi.importances_std):
            rows.append({"metode": f"permutasi_{best}_val", "fitur": f, "importance": mu, "std": sd})
        imp = pd.DataFrame(rows)
        imp["kelompok"] = imp["fitur"].map(_group)
        imp.to_csv(out_tables / f"04_importance_{exp}.csv", index=False)
        top = imp[imp.metode.str.startswith("permutasi")].nlargest(5, "importance")["fitur"].tolist()
        log.info(f"Top-5 permutation importance ({best}): {top}")

    # ---------------- ablasi fitur ----------------
    if not args.skip_ablation:
        imb = cand[best]["imbalance"]
        rows = []
        for name, cols in tqdm(mm.feature_subsets(feats).items(), desc=f"ablasi {exp}"):
            est, summ = tune(best, imb, Xtr[cols], ytr, cols, cv, args.n_iter_ablation, args.n_jobs, seed)
            pva, pte = proba(est, Xva[cols]), proba(est, Xte[cols])
            thr = ev.youden_threshold(yva, pva)
            m = ev.compute_metrics(yte, pte, thr, n_boot=args.n_boot, seed=seed)
            rows.append({"subset": name, "n_fitur": len(cols), "model": mm.NAMA_MODEL[best], "imbalance": imb,
                         "cv_auc_mean": summ["cv_auc_mean"], "cv_auc_std": summ["cv_auc_std"],
                         "val_auc": roc_auc_score(yva, pva), "test_auc": m["roc_auc"],
                         "test_auc_ci_low": m["roc_auc_ci_low"], "test_auc_ci_high": m["roc_auc_ci_high"],
                         "test_pr_auc": m["pr_auc"], "test_sensitivity": m["recall_sensitivity"],
                         "test_specificity": m["specificity"], "test_f1": m["f1"]})
            log.info(f"[ablasi] {name:28s} n={len(cols):3d} CV {summ['cv_auc_mean']:.3f} | "
                     f"val {rows[-1]['val_auc']:.3f} | test {m['roc_auc']:.3f}")
        fn = "04_ablasi_fitur.csv" if exp == "A" else f"04_ablasi_fitur_{exp}.csv"
        pd.DataFrame(rows).to_csv(out_tables / fn, index=False)

    with open((cfg.paths.logs if not limit_mode else out_tables) / f"train_ml_{exp}_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=float)


def _group(c: str) -> str:
    from src.features import feature_group
    return feature_group(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=["A", "B"])
    ap.add_argument("--limit", type=int, default=None, help="subsampel N baris per split (uji cepat)")
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--n-iter-ablation", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-2)
    ap.add_argument("--overwrite", action="store_true", help="latih ulang walau model sudah ada")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-importance", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    for exp in args.experiments:
        log = setup_logger(cfg.paths.logs / (f"train_ml_{exp}.log" if args.limit is None
                                             else f"train_ml_{exp}_uji.log"))
        t0 = time.time()
        run_experiment(exp, args, cfg, log)
        log.info(f"Eksperimen {exp} selesai dalam {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
