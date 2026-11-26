"""Tahap 2 — penetapan dataset eksperimen + preprocessing audio COUGHVID.

Langkah:
  1. Bangun tabel label per eksperimen (src.data_utils.build_label_table).
  2. Split STRATIFIED 70/15/15 per uuid (seed 42) → data/processed/splits_<X>.csv (FINAL; tidak
     ditimpa kecuali --resplit). Kolom `dipakai` menandai undersampling healthy di train.
  3. Preprocessing paralel (joblib) untuk gabungan uuid yang dipakai semua eksperimen:
     konversi → trim → segmentasi → reduksi noise → normalisasi (src.preprocessing).
     Resume: uuid yang sudah tercatat di data/processed/preprocess_meta.csv dilewati.
  4. Tulis manifest_<X>.csv, segments.csv, reports/logs/preprocess_errors.csv & ringkasan.

Contoh:
  python scripts/02_preprocess.py --limit 50          # uji cepat
  python scripts/02_preprocess.py                     # run penuh (melanjutkan bila terputus)
  python scripts/02_preprocess.py --experiments A B C
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_utils as du  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.config import load_config, set_seed  # noqa: E402

META_COLS = ["uuid", "path_wav", "sr", "durasi_asli_s", "durasi_trim_s", "trim_mulai_s", "n_segmen",
             "durasi_batuk_s", "segments", "status", "error", "src_path"]


def _rel(p: str | Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def _worker(uuid: str, src: str, overwrite: bool) -> dict:
    """Bungkus preprocess_file agar error per file tidak menghentikan proses."""
    import warnings

    warnings.filterwarnings("ignore")
    try:
        r = pp.preprocess_file(uuid, src, overwrite=overwrite)
        r["path_wav"] = _rel(r["path_wav"])
        r["segments"] = json.dumps(r["segments"])
    except pp.PreprocessError as e:
        r = {"uuid": uuid, "status": "dibuang", "error": str(e)}
    except Exception as e:  # file rusak / error tak terduga
        r = {"uuid": uuid, "status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
    r["src_path"] = _rel(src)
    return r


def build_splits(df: pd.DataFrame, exp: str, cfg, resplit: bool, log: logging.Logger) -> pd.DataFrame:
    path = cfg.paths.processed / f"splits_{exp}.csv"
    labels, stats = du.build_label_table(df, exp, cfg, return_stats=True)
    log.info("Eksperimen %s — statistik label: %s", exp, stats)
    (cfg.paths.logs / f"label_stats_{exp}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if path.exists() and not resplit:
        sp = pd.read_csv(path)
        if set(sp["uuid"]) != set(labels["uuid"]):
            raise RuntimeError(f"{path.name} tidak cocok dengan populasi saat ini; pakai --resplit bila disengaja")
        log.info("Memakai split FINAL yang sudah ada: %s", path.name)
        return sp
    sp = du.make_splits(labels, cfg)
    ratio = cfg.data.get("healthy_to_covid_ratio") if exp == "A" else None
    sp["dipakai"] = du.undersample_train(sp, ratio, seed=cfg.seed)
    sp = sp.drop(columns=["audio_path"])
    sp.to_csv(path, index=False)
    log.info("Split disimpan: %s (rasio undersampling train: %s)", path.name, ratio)
    return sp


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="proses hanya N rekaman (uji cepat)")
    ap.add_argument("--experiments", nargs="+", default=None, help="default: data.experiments di config")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--resplit", action="store_true", help="buat ulang split (HATI-HATI: split seharusnya final)")
    ap.add_argument("--retry-failed", action="store_true", help="ulangi rekaman yang sebelumnya gagal")
    ap.add_argument("--overwrite", action="store_true", help="proses ulang semua file")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    for k in ("processed", "interim", "logs"):
        cfg.paths[k].mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.paths.logs / "preprocess_run.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("prep")
    exps = [e.upper() for e in (args.experiments or cfg.data.experiments)]
    n_jobs = args.n_jobs or cfg.preprocessing.n_jobs
    log.info("=== Tahap 2 preprocessing | eksperimen=%s limit=%s n_jobs=%s ===", exps, args.limit, n_jobs)

    df = du.load_metadata()
    src_map = df.set_index("uuid")["audio_path"]
    splits = {e: build_splits(df, e, cfg, args.resplit, log) for e in exps}

    # --- daftar pekerjaan: gabungan uuid yang dipakai ---
    uuids = sorted(set().union(*[set(s.loc[s["dipakai"], "uuid"]) for s in splits.values()]))
    meta_path = cfg.paths.processed / "preprocess_meta.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() and not args.overwrite else pd.DataFrame(columns=META_COLS)
    meta = meta.drop_duplicates("uuid", keep="last")
    if args.retry_failed:
        meta = meta[meta["status"] == "ok"]
    done_ok = set(meta.loc[meta["status"] == "ok", "uuid"])
    # file ok tapi wav hilang → proses ulang
    missing = {u for u in done_ok if not (cfg.paths.processed / "wav" / f"{u}.wav").exists()}
    meta = meta[~meta["uuid"].isin(missing)]
    done = set(meta["uuid"])
    todo = [u for u in uuids if u not in done]
    if args.limit:
        todo = pd.Series(todo).sample(n=min(args.limit, len(todo)), random_state=cfg.seed).tolist()
    log.info("Total uuid dipakai: %d | sudah diproses: %d | akan diproses: %d", len(uuids),
             len(done & set(uuids)), len(todo))

    # --- preprocessing paralel ---
    t0 = time.time()
    new_rows = []
    if todo:
        jobs = (delayed(_worker)(u, src_map[u], args.overwrite) for u in todo)
        gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered", batch_size=4)(jobs)
        with open(meta_path, "a", encoding="utf-8", newline="") as fh:
            if fh.tell() == 0 or args.overwrite:
                fh.seek(0); fh.truncate()
                pd.DataFrame(columns=META_COLS).to_csv(fh, index=False)
            for i, r in enumerate(tqdm(gen, total=len(todo), desc="preprocess", unit="file")):
                row = pd.DataFrame([{c: r.get(c, "") for c in META_COLS}])
                row.to_csv(fh, header=False, index=False)
                new_rows.append(r)
                if i % 100 == 0:
                    fh.flush()
    elapsed = time.time() - t0
    log.info("Selesai memproses %d file dalam %.1f s (%.3f s/file)", len(todo), elapsed,
             elapsed / max(1, len(todo)))

    # --- konsolidasi ---
    meta = pd.read_csv(meta_path).drop_duplicates("uuid", keep="last")
    meta.to_csv(meta_path, index=False)  # rapikan (hapus duplikat dari retry)
    err = meta[meta["status"] != "ok"][["uuid", "src_path", "status", "error"]]
    err.to_csv(cfg.paths.logs / "preprocess_errors.csv", index=False)
    log.info("File gagal/dibuang: %d → reports/logs/preprocess_errors.csv", len(err))
    if len(err):
        log.info("Ringkasan error:\n%s", err["error"].str.replace(r"\(.*", "", regex=True).value_counts().head(10))

    ok = meta[meta["status"] == "ok"].copy()
    seg_rows = [{"uuid": u, "seg_idx": i, "start_s": a, "end_s": b}
                for u, s in zip(ok["uuid"], ok["segments"]) for i, (a, b) in enumerate(json.loads(s))]
    pd.DataFrame(seg_rows, columns=["uuid", "seg_idx", "start_s", "end_s"]).to_csv(
        cfg.paths.processed / "segments.csv", index=False)

    summary = {"waktu_run_s": round(elapsed, 1), "n_diproses_run_ini": len(todo), "limit": args.limit,
               "n_ok_total": int(len(ok)), "n_gagal_total": int(len(err))}
    mcols = ["uuid", "path_wav", "label", "label_name", "split", "durasi_asli_s", "durasi_trim_s",
             "n_segmen", "durasi_batuk_s", "sr", "cough_detected", "status"]
    for e, sp in splits.items():
        used = sp[sp["dipakai"]]
        m = used.merge(ok.drop(columns=["status"]), on="uuid", how="inner")
        extra = ["diagnosis_vote", "n_ahli", "ada_konflik"] if e == "B" else []
        m = m[mcols + extra].rename(columns={"status": "status_self_reported"})
        m.to_csv(cfg.paths.processed / f"manifest_{e}.csv", index=False)
        tab = pd.crosstab(m["split"], m["label_name"], margins=True, margins_name="total")
        summary[f"manifest_{e}"] = {"n": int(len(m)), "belum_diproses": int(len(used) - len(m) -
                                    used["uuid"].isin(err["uuid"]).sum()),
                                    "gagal": int(used["uuid"].isin(err["uuid"]).sum()),
                                    "per_split": tab.to_dict()}
        log.info("manifest_%s.csv: %d baris\n%s", e, len(m), tab)
    (cfg.paths.logs / "preprocess_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                            encoding="utf-8")


if __name__ == "__main__":
    main()
