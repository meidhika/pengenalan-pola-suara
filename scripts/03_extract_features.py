"""Tahap 3 — ekstraksi fitur tabular & 2D dari wav hasil Tahap 2.

Langkah:
  1. Gabungkan uuid dari manifest_<X>.csv (eksperimen di config `data.experiments`); fitur tidak
     bergantung eksperimen, jadi uuid yang ada di A dan B cukup dihitung sekali.
  2. Ekstraksi paralel (ProcessPool) → cache per rekaman data/features/cache/<uuid>.npz.
     RESUME: uuid yang cache-nya sudah ada dilewati (kecuali --overwrite).
  3. Rakit keluaran per eksperimen:
       data/features/tabular_<X>.parquet         (uuid, split, label, label_name, + fitur)
       data/features/logmel_<X>_<split>.npy      (N, 64, 1001) float16
       data/features/mfcc2d_<X>_<split>.npy      (N, 40, 1001) float16
       data/features/labels_<X>_<split>.npy      (N,) int8   — urutan sama dengan array fitur
       data/features/uuids_<X>_<split>.npy       (N,) str
  4. Cek NaN/inf → reports/logs/features_nan_report.csv; ringkasan → reports/logs/features_summary.json.

Contoh:
  python scripts/03_extract_features.py --limit 50     # uji cepat (keluaran ke data/features/uji_limit/)
  python scripts/03_extract_features.py                # run penuh (melanjutkan bila terputus)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Hindari oversubscription thread BLAS/FFT di setiap proses worker
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import features as fx  # noqa: E402
from src.config import load_config, set_seed  # noqa: E402

SPLITS = ("train", "val", "test")


def _worker(uuid: str, wav: str, cache_dir: str, overwrite: bool) -> dict:
    try:
        return fx.extract_file(uuid, ROOT / wav, cache_dir, overwrite=overwrite)
    except Exception as e:  # file bermasalah tidak menghentikan proses
        return {"uuid": uuid, "status": "error", "error": f"{type(e).__name__}: {e}"[:300]}


def _n_jobs(n: int) -> int:
    c = os.cpu_count() or 1
    return max(1, c + 1 + n if n < 0 else n)


def setup_logger(path: Path) -> logging.Logger:
    log = logging.getLogger("fitur")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def assemble(exp: str, man: pd.DataFrame, cache_dir: Path, out_dir: Path, cfg, log) -> dict:
    """Rakit parquet tabular + array 2D per split untuk satu eksperimen."""
    man = man[man["uuid"].map(lambda u: (cache_dir / f"{u}.npz").exists())].reset_index(drop=True)
    names, rows = None, []
    dt = np.dtype(cfg.features.dtype_2d)
    info = {"n_rekaman": len(man), "split": {}}
    for split in SPLITS:
        sub = man[man["split"] == split].reset_index(drop=True)
        shp = fx.shape_2d(cfg)
        # memmap: array besar ditulis langsung ke disk (hemat RAM); split kosong → array biasa
        def _arr(kind):
            p, s = out_dir / f"{kind}_{exp}_{split}.npy", (len(sub), *shp[kind])
            return np.lib.format.open_memmap(p, mode="w+", dtype=dt, shape=s) if len(sub) else np.zeros(s, dt)
        L, M = _arr("logmel"), _arr("mfcc2d")
        n_bad_2d = 0
        for i, r in enumerate(tqdm(sub.itertuples(), total=len(sub), desc=f"rakit {exp}/{split}", leave=False)):
            z = fx.load_cache(r.uuid, cache_dir)
            if names is None:
                names = [str(s) for s in z["names"]]
            rows.append(z["values"])
            L[i], M[i] = z["logmel"], z["mfcc2d"]
            n_bad_2d += int(not (np.isfinite(z["logmel"]).all() and np.isfinite(z["mfcc2d"]).all()))
        if len(sub):
            L.flush(), M.flush()
        else:
            np.save(out_dir / f"logmel_{exp}_{split}.npy", L), np.save(out_dir / f"mfcc2d_{exp}_{split}.npy", M)
        del L, M
        np.save(out_dir / f"labels_{exp}_{split}.npy", sub["label"].to_numpy(np.int8))
        np.save(out_dir / f"uuids_{exp}_{split}.npy", sub["uuid"].to_numpy(str))
        info["split"][split] = {"n": len(sub), "n_covid": int(sub["label"].sum()),
                                "shape_logmel": [len(sub), *shp["logmel"]],
                                "shape_mfcc2d": [len(sub), *shp["mfcc2d"]], "n_2d_tidak_finite": n_bad_2d}
        if n_bad_2d:
            log.warning("%s/%s: %d array 2D berisi NaN/inf", exp, split, n_bad_2d)

    order = pd.concat([man[man["split"] == s] for s in SPLITS], ignore_index=True)
    tab = pd.DataFrame(np.vstack(rows), columns=names)
    tab = pd.concat([order[["uuid", "split", "label", "label_name"]].reset_index(drop=True), tab], axis=1)
    tab, rep = fx.clean_tabular(tab, names)
    tab.to_parquet(out_dir / f"tabular_{exp}.parquet", index=False)
    rep.insert(0, "eksperimen", exp)
    info.update({"n_fitur_tabular": len(names), "shape_tabular": list(tab.shape),
                 "n_kolom_ada_nan_inf": int(len(rep)),
                 "n_rekaman_tanpa_voiced": int((tab["voiced_ratio"] == 0).sum()),
                 "fitur_per_kelompok": pd.Series([fx.feature_group(c) for c in names]).value_counts().to_dict()})
    return info, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="proses N uuid pertama saja (uji cepat)")
    ap.add_argument("--overwrite", action="store_true", help="hitung ulang walau cache ada")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--experiments", nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    feat_dir = cfg.paths.features
    cache_dir = feat_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    log = setup_logger(cfg.paths.logs / "features_run.log")
    exps = args.experiments or list(cfg.data.experiments)
    log.info("=== Tahap 3: ekstraksi fitur | eksperimen %s | limit %s ===", exps, args.limit)

    mans = {e: pd.read_csv(cfg.paths.processed / f"manifest_{e}.csv") for e in exps}
    allu = pd.concat([m[["uuid", "path_wav"]] for m in mans.values()]).drop_duplicates("uuid")
    allu = allu.sort_values("uuid").reset_index(drop=True)
    if args.limit:
        # uji cepat: ambil campuran kedua kelas dari manifest pertama agar uji statistik bermakna
        m0 = mans[exps[0]]
        pick = pd.concat([m0[m0.label == 1].head(args.limit // 2), m0[m0.label == 0].head(args.limit - args.limit // 2)])
        allu = allu[allu.uuid.isin(pick.uuid)].reset_index(drop=True)
        out_dir = feat_dir / "uji_limit"
    else:
        out_dir = feat_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = allu if args.overwrite else allu[~allu.uuid.map(lambda u: (cache_dir / f"{u}.npz").exists())]
    log.info("uuid total %d | sudah ada cache %d | akan diproses %d", len(allu), len(allu) - len(todo), len(todo))

    t0 = time.time()
    errors = []
    if len(todo):
        nj = _n_jobs(args.n_jobs if args.n_jobs is not None else cfg.features.n_jobs)
        log.info("worker: %d", nj)
        with ProcessPoolExecutor(max_workers=nj) as ex:
            futs = [ex.submit(_worker, r.uuid, r.path_wav, str(cache_dir), args.overwrite) for r in todo.itertuples()]
            for f in tqdm(as_completed(futs), total=len(futs), desc="ekstraksi", mininterval=2):
                r = f.result()
                if r["status"] == "error":
                    errors.append(r)
                    log.error("%s: %s", r["uuid"], r["error"])
    t_ext = time.time() - t0
    log.info("ekstraksi selesai: %.1f s (%d file, %d error)", t_ext, len(todo), len(errors))
    if errors:
        pd.DataFrame(errors).to_csv(cfg.paths.logs / "features_errors.csv", index=False)

    summary = {"waktu_ekstraksi_s": round(t_ext, 1), "n_diproses": len(todo), "n_error": len(errors),
               "parameter": dict(cfg.features), "eksperimen": {}}
    reps = []
    for e, man in mans.items():
        if args.limit:
            man = man[man.uuid.isin(allu.uuid)]
        info, rep = assemble(e, man, cache_dir, out_dir, cfg, log)
        summary["eksperimen"][e] = info
        reps.append(rep)
        log.info("Eksperimen %s: tabular %s | %s", e, info["shape_tabular"],
                 {s: v["shape_logmel"] for s, v in info["split"].items()})
    rep = pd.concat(reps)
    tag = "_uji_limit" if args.limit else ""
    rep.to_csv(cfg.paths.logs / f"features_nan_report{tag}.csv")
    (cfg.paths.logs / f"features_summary{tag}.json").write_text(json.dumps(summary, indent=2, default=str),
                                                                 encoding="utf-8")
    log.info("Kolom dengan NaN/inf (sebelum ditangani): %d baris laporan. Selesai dalam %.1f s.",
             len(rep), time.time() - t0)


if __name__ == "__main__":
    main()
