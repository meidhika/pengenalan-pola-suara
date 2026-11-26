"""Tahap 5 — Training CNN ringan pada fitur 2D (log-mel / MFCC 2D), Eksperimen A (default).

Protokol (sama dengan ML klasik Tahap 4 agar sebanding):
  1. Normalisasi per pita frekuensi di-fit pada TRAIN saja; augmentasi hanya pada train.
  2. Loss BCEWithLogits dengan pos_weight = n_healthy/n_COVID di train (lihat alasan di bawah).
  3. AdamW (lr 1e-3, weight decay 1e-2), ReduceLROnPlateau pada AUC val (faktor 0,5, patience 4),
     maks 100 epoch, early stopping pada AUC val (patience 10) → checkpoint terbaik.
  4. Ambang Youden dipilih di VAL dengan checkpoint terbaik; TEST dievaluasi sekali.
  5. Diulang untuk seed 42, 7, 2024 → mean ± std AUC.

Mengapa pos_weight, bukan WeightedRandomSampler:
  - Train A sudah di-undersample 3:1 (Tahap 2) sehingga imbalance tinggal sedang; pos_weight ≈ 3 cukup.
  - Sampler dengan penggantian mengulang ±500 rekaman COVID-19 sekitar 3× per epoch → risiko hafalan
    (overfitting) lebih tinggi, dan sebagian rekaman healthy tidak terlihat pada epoch tertentu.
  - pos_weight deterministik, setiap rekaman dilihat tepat sekali per epoch, setara dengan
    class_weight="balanced" yang dipakai model ML klasik → perbandingan lebih adil.
  Catatan: pos_weight (seperti undersampling) menggeser kalibrasi probabilitas → ambang 0,5 tidak
  bermakna; karena itu ambang selalu dipilih di VAL (Youden).

Keluaran (run penuh):
  models/cnn_<fitur>[<tag>]_<X>.pt              (seed 42; seed lain: ..._seed<s>.pt)
  reports/logs/cnn_<fitur>[<tag>]_history.csv   (semua seed; kolom `seed`)
  reports/tables/05_cnn_per_seed.csv            (metrik val & test per seed)
  reports/tables/hasil_semua_model.csv          (1 baris per fitur: seed 42 + kolom mean/std 3 seed)
  models/preds_cnn_<X>.parquet                  (skor val/test tiap run → kurva ROC bersama ML)
  reports/logs/train_cnn_<X>.log, train_cnn_<X>_summary.json

Contoh:
  python scripts/05_train_cnn.py --limit 200 --epochs 2 --seeds 42      # uji cepat
  python scripts/05_train_cnn.py --features logmel mfcc2d                # run penuh 3 seed
  python scripts/05_train_cnn.py --features logmel --kernel 1x12 --tag _k1x12 --seeds 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src import models_cnn as mc  # noqa: E402
from src.config import load_config, set_seed  # noqa: E402

PER_SEED_TABLE = "05_cnn_per_seed.csv"


def setup_logger(path: Path) -> logging.Logger:
    log = logging.getLogger("cnn")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    if hasattr(sys.stdout, "reconfigure"):  # konsol Windows (cp1252) → UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="A", choices=["A", "B"])
    p.add_argument("--features", nargs="+", default=["logmel", "mfcc2d"], choices=list(mc.FITUR_2D))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024])
    p.add_argument("--arch", default="LightCNN", choices=["LightCNN", "CRNN"])
    p.add_argument("--kernel", default="3x3", help="3x3 atau 1x12 (ala R3)")
    p.add_argument("--n-blocks", type=int, default=4, choices=[4, 5])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=None, help="default: 32 (CPU) / 64 (CUDA)")
    p.add_argument("--time-pool", type=int, default=2, help="rata-rata tiap k frame (10 ms → 20 ms)")
    p.add_argument("--crop", type=int, default=256, help="panjang random crop train (frame setelah pooling)")
    p.add_argument("--mixup", type=float, default=0.0, help="alpha mixup; 0 = nonaktif")
    p.add_argument("--no-specaugment", action="store_true")
    p.add_argument("--tag", default="", help="akhiran nama run untuk varian, mis. _k1x12")
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (CPU)")
    p.add_argument("--limit", type=int, default=None, help="subsampel stratified per split (uji cepat)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def stratified_limit(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indeks subsampel stratified (minimal 5 positif) — hanya untuk --limit."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = min(len(pos), max(5, int(round(n * len(pos) / len(y)))))
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, min(len(neg), n - n_pos), replace=False)])
    return np.sort(idx)


def model_label(feature: str, args) -> str:
    """Nama model di tabel hasil, mis. 'CNN (log-mel)', 'CNN (log-mel, kernel 1x12)', 'CRNN (MFCC 2D)'."""
    if args.arch == "CRNN":
        return f"CRNN ({mc.FITUR_2D[feature]})"
    extra = "" if args.kernel == "3x3" else f", kernel {args.kernel}"
    tag = args.tag if (args.tag and not extra) else ""
    return f"CNN ({mc.FITUR_2D[feature]}{extra}){tag}"


def load_feature(feature, args, log):
    """Muat & normalisasi train/val/test untuk satu jenis fitur (sekali untuk semua seed)."""
    data = {}
    for split in ["train", "val", "test"]:
        y_all = np.load(load_config().paths.features / f"labels_{args.experiment}_{split}.npy")
        idx = None
        if args.limit:
            idx = stratified_limit(y_all, min(args.limit, len(y_all)), 42)
        data[split] = mc.load_split(feature, args.experiment, split, args.time_pool, idx)
        X, y, _, L = data[split]
        log.info(f"  {split:5s}: X{X.shape} pos={int(y.sum())} neg={int((y == 0).sum())} "
                 f"panjang isi median={int(np.median(L))} frame")
    norm = mc.BandNormalizer().fit(data["train"][0])
    for split in data:
        X, y, u, L = data[split]
        data[split] = (norm.transform(X), y, u, L)
    return data, norm


def run_one(feature, seed, data, norm, args, paths, device, log) -> dict:
    exp = args.experiment
    run = f"cnn_{feature}{args.tag}_{exp}"
    ckpt = paths["models"] / (f"{run}.pt" if seed == 42 else f"{run}_seed{seed}.pt")
    last = ckpt.with_suffix(".last.pt")

    if ckpt.exists() and not args.overwrite:
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        if st.get("finished"):
            log.info(f"[{run} seed {seed}] sudah selesai → dimuat dari {ckpt.name} (pakai --overwrite untuk ulang)")
            return st["summary"] | {"_preds": st["preds"], "_history": pd.DataFrame(st["history"])}

    set_seed(seed)
    Xtr, ytr, _, Ltr = data["train"]
    n_freq = Xtr.shape[1]
    sa = dict(n_freq_masks=0, n_time_masks=0) if args.no_specaugment else {}
    fmask = 8 if n_freq >= 64 else 5
    ds_tr = mc.SpecDataset(Xtr, ytr, Ltr, train=True, crop=args.crop, freq_mask=fmask, seed=seed, **sa)
    gen = torch.Generator().manual_seed(seed)
    bs = args.batch_size or mc.default_batch_size(device)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, generator=gen, drop_last=True, num_workers=0)
    dl_ev = {s: DataLoader(mc.SpecDataset(data[s][0], data[s][1]), batch_size=bs * 2, shuffle=False)
             for s in ["val", "test"]}

    if args.arch == "LightCNN":
        cfg = dict(arch="LightCNN", n_blocks=args.n_blocks, kernel=args.kernel, dropout=args.dropout)
    else:
        cfg = dict(arch="CRNN", n_freq=n_freq, dropout=args.dropout)
    model = mc.build_model(cfg).to(device)
    n_params = mc.count_params(model)
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4, min_lr=1e-5)

    history, best_auc, best_state, best_epoch, wait, start_epoch, t_train = [], -1.0, None, 0, 0, 1, 0.0
    if last.exists() and not args.overwrite:  # lanjutkan run yang terputus
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        history, best_auc, best_state = st["history"], st["best_auc"], st["best_state"]
        best_epoch, wait, start_epoch, t_train = st["best_epoch"], st["wait"], st["epoch"] + 1, st["t_train"]
        torch.set_rng_state(st["torch_rng"]); gen.set_state(st["gen_state"]); ds_tr.rng = st["ds_rng"]
        log.info(f"[{run} seed {seed}] lanjut dari epoch {start_epoch}")

    log.info(f"[{run} seed {seed}] {cfg} | param={n_params:,} | pos_weight={pos_weight:.2f} | batch={bs} "
             f"| device={device}")
    yv = data["val"][1]
    for epoch in range(start_epoch, args.epochs + 1):
        if wait >= args.patience:
            break
        t0 = time.time()
        model.train()
        losses, logits_tr, y_tr = [], [], []
        for xb, yb in tqdm(dl_tr, desc=f"{feature} s{seed} ep{epoch}", leave=False, ncols=90):
            xb, yb = xb.to(device), yb.to(device)
            if args.mixup > 0:
                xb, yb = mc.mixup_batch(xb, yb, args.mixup, gen)
            out = model(xb)
            loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            logits_tr.append(out.detach().cpu().numpy()); y_tr.append(yb.cpu().numpy())
        y_tr = np.concatenate(y_tr)
        tr_auc = roc_auc_score(y_tr, np.concatenate(logits_tr)) if args.mixup == 0 else np.nan

        # validasi (rekaman penuh, tanpa augmentasi)
        model.eval()
        v_logits, v_losses = [], []
        with torch.no_grad():
            for xb, yb in dl_ev["val"]:
                o = model(xb.to(device))
                v_losses.append(crit(o, yb.to(device)).item() * len(yb))
                v_logits.append(o.cpu().numpy())
        v_logits = np.concatenate(v_logits)
        val_auc = roc_auc_score(yv, v_logits)
        val_loss = float(np.sum(v_losses) / len(yv))
        sched.step(val_auc)
        dt = time.time() - t0
        t_train += dt
        improved = val_auc > best_auc
        if improved:
            best_auc, best_epoch, wait = val_auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        history.append(dict(seed=seed, epoch=epoch, train_loss=float(np.mean(losses)), train_auc=float(tr_auc),
                            val_loss=val_loss, val_auc=float(val_auc), lr=opt.param_groups[0]["lr"],
                            detik=round(dt, 1), terbaik=improved))
        log.info(f"  ep{epoch:3d} loss {np.mean(losses):.4f} aucTr {tr_auc:.3f} | val loss {val_loss:.4f} "
                 f"auc {val_auc:.3f}{' *' if improved else ''} | lr {opt.param_groups[0]['lr']:.1e} | {dt:.0f}s")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), history=history,
                        best_auc=best_auc, best_state=best_state, best_epoch=best_epoch, wait=wait, epoch=epoch,
                        t_train=t_train, torch_rng=torch.get_rng_state(), gen_state=gen.get_state(),
                        ds_rng=ds_tr.rng), last)

    # evaluasi akhir dengan checkpoint terbaik
    model.load_state_dict(best_state)
    p_val = mc.predict_proba(model, dl_ev["val"], device)
    p_test = mc.predict_proba(model, dl_ev["test"], device)
    thr = ev.youden_threshold(yv, p_val)
    m_val = ev.compute_metrics(yv, p_val, thr, seed=42)
    m_test = ev.compute_metrics(data["test"][1], p_test, thr, seed=42)
    hist = pd.DataFrame(history)
    last_ep = hist.iloc[-1]
    best_row = hist.loc[hist.epoch == best_epoch].iloc[0]
    summary = dict(feature=feature, seed=seed, run=run, arch=args.arch, kernel=args.kernel,
                   n_params=n_params, pos_weight=pos_weight, best_epoch=int(best_epoch),
                   n_epochs=int(hist.epoch.max()), train_time_s=round(t_train, 1),
                   train_auc_best=float(best_row.train_auc), val_auc=float(m_val["roc_auc"]),
                   test_auc=float(m_test["roc_auc"]), threshold=float(thr),
                   train_loss_last=float(last_ep.train_loss), val_loss_last=float(last_ep.val_loss),
                   val_loss_min=float(hist.val_loss.min()))
    preds = dict(val=dict(uuid=data["val"][2], label=yv, score=p_val),
                 test=dict(uuid=data["test"][2], label=data["test"][1], score=p_test))
    torch.save(dict(model=best_state, config=cfg, normalizer=norm.state(), time_pool=args.time_pool,
                    feature=feature, experiment=exp, seed=seed, threshold=thr, history=history,
                    summary=summary, metrics_val=m_val, metrics_test=m_test, preds=preds,
                    args=vars(args), finished=True), ckpt)
    last.unlink(missing_ok=True)

    label = model_label(feature, args)
    extra = dict(imbalance="pos_weight", seed=seed, val_auc=m_val["roc_auc"], n_params=n_params,
                 best_epoch=best_epoch, n_epochs=summary["n_epochs"], train_time_s=summary["train_time_s"])
    ev.save_results(f"{label} s{seed}", exp, m_test | extra, "test", out=paths["tables"] / PER_SEED_TABLE)
    ev.save_results(f"{label} s{seed}", exp, m_val | extra, "val", out=paths["tables"] / PER_SEED_TABLE)
    log.info(f"[{run} seed {seed}] selesai: best ep {best_epoch}/{summary['n_epochs']} | AUC val "
             f"{m_val['roc_auc']:.3f} | AUC test {m_test['roc_auc']:.3f} "
             f"[{m_test['roc_auc_ci_low']:.3f}–{m_test['roc_auc_ci_high']:.3f}] | thr {thr:.3f} | "
             f"{t_train / 60:.1f} menit")
    return summary | {"_preds": preds, "_history": hist}


def save_preds(results, path: Path, label_col: str):
    """Upsert kolom skor per run ke parquet prediksi (val+test)."""
    frames = []
    for r in results:
        col = f"{label_col(r)}_s{r['seed']}"
        for split, d in r["_preds"].items():
            frames.append(pd.DataFrame({"uuid": d["uuid"], "label": d["label"], "split": split, col: d["score"]}))
    if not frames:
        return
    new = None
    for col in {c for f in frames for c in f.columns if c not in ("uuid", "label", "split")}:
        part = pd.concat([f for f in frames if col in f.columns])
        new = part if new is None else new.merge(part, on=["uuid", "label", "split"], how="outer")
    if path.exists():
        old = pd.read_parquet(path)
        old = old.drop(columns=[c for c in new.columns if c not in ("uuid", "label", "split") and c in old.columns])
        new = old.merge(new, on=["uuid", "label", "split"], how="outer")
    new.to_parquet(path, index=False)


def main():
    args = parse_args()
    cfg = load_config()
    device = mc.get_device()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.limit:
        base = cfg.paths.logs / "uji_limit"
        paths = dict(models=base, tables=base, logs=base)
    else:
        paths = dict(models=cfg.paths.models, tables=cfg.paths.tables, logs=cfg.paths.logs)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    log = setup_logger(paths["logs"] / f"train_cnn_{args.experiment}.log")
    log.info(f"=== Tahap 5 CNN | eksperimen {args.experiment} | {vars(args)}")
    log.info(f"device={device} | torch {torch.__version__} | threads={torch.get_num_threads()}")

    summary_path = paths["logs"] / f"train_cnn_{args.experiment}_summary.json"
    all_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    t_all = time.time()
    for feature in args.features:
        log.info(f"--- fitur {feature}: memuat data")
        data, norm = load_feature(feature, args, log)
        results = [run_one(feature, s, data, norm, args, paths, device, log) for s in args.seeds]

        # riwayat per epoch (semua seed) → CSV
        sfx = "" if args.experiment == "A" else f"_{args.experiment}"
        hist_path = paths["logs"] / f"cnn_{feature}{args.tag}{sfx}_history.csv"
        hist_new = pd.concat([r["_history"] for r in results])
        if hist_path.exists():
            old = pd.read_csv(hist_path)
            hist_new = pd.concat([old[~old.seed.isin(args.seeds)], hist_new])
        hist_new.sort_values(["seed", "epoch"]).to_csv(hist_path, index=False)

        # prediksi
        arch_short = "cnn" if args.arch == "LightCNN" else "crnn"
        save_preds(results, paths["models"] / f"preds_cnn_{args.experiment}.parquet",
                   lambda r: f"{arch_short}_{r['feature']}{args.tag}")

        # ringkasan mean ± std antar seed
        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
        agg = {f"{c}_mean": df[c].mean() for c in ["val_auc", "test_auc", "best_epoch", "train_time_s"]}
        agg |= {f"{c}_std": df[c].std(ddof=1) if len(df) > 1 else 0.0 for c in ["val_auc", "test_auc"]}
        key = f"{args.arch}_{feature}{args.tag}_{args.experiment}"
        all_summary[key] = dict(seeds=args.seeds, per_seed=df.to_dict("records"), **agg)
        log.info(f"=== {key}: AUC test {agg['test_auc_mean']:.3f} ± {agg['test_auc_std']:.3f} | "
                 f"AUC val {agg['val_auc_mean']:.3f} ± {agg['val_auc_std']:.3f} (n seed = {len(df)})")

        # baris utama di hasil_semua_model.csv: seed 42 (ditetapkan sebelum melihat hasil) + mean/std seed
        if 42 in args.seeds:
            st = torch.load(paths["models"] / f"cnn_{feature}{args.tag}_{args.experiment}.pt",
                            map_location="cpu", weights_only=False)
            row = st["metrics_test"] | dict(
                imbalance="pos_weight", val_auc=st["metrics_val"]["roc_auc"],
                cnn_seed_auc_mean=agg["test_auc_mean"], cnn_seed_auc_std=agg["test_auc_std"],
                cnn_seed_val_auc_mean=agg["val_auc_mean"], cnn_seed_val_auc_std=agg["val_auc_std"],
                n_params=st["summary"]["n_params"], best_epoch=st["summary"]["best_epoch"],
                train_time_s=st["summary"]["train_time_s"],
                best_params=json.dumps({k: v for k, v in st["config"].items()} | {"seeds": args.seeds}))
            out = None if not args.limit else paths["tables"] / ev.RESULTS_FILE
            ev.save_results(model_label(feature, args), args.experiment, row, "test", out=out)

    summary_path.write_text(json.dumps(all_summary, indent=2, default=float), encoding="utf-8")
    log.info(f"Total waktu: {(time.time() - t_all) / 60:.1f} menit")


if __name__ == "__main__":
    main()
