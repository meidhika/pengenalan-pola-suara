"""Tahap 6 — Evaluasi komprehensif, perbandingan, dan analisis (TANPA training ulang).

Memakai model & prediksi tersimpan di models/ (Tahap 4: ML klasik A/B, Tahap 5: CNN A).

Langkah:
  a. Tabel final Eksperimen A (+ B sebagai pembanding): semua metrik test + penanda model terbaik.
  b. Uji signifikansi selisih AUC: DeLong + bootstrap berpasangan (dua model teratas & pasangan kunci).
  c. ROC/PR gabungan: ML terbaik vs CNN vs baseline.
  d. Eksperimen B vs A + uji silang label (model A → data berlabel ahli, model B → test A) dengan
     MEMBUANG uuid yang dipakai saat training/validasi model sumber (anti leakage) + kesepakatan label.
  e. Analisis error: distribusi skor per kelas, contoh FP/FN (log-mel), karakteristik error, subgrup.
  f. Tabel perbandingan dengan literatur (angka literatur hanya dari PDF di references/; ragu → TODO).

Keluaran: reports/tables/06_*.csv|md, reports/figures/06_*.png, reports/logs/evaluate_all.log + _summary.json.

Contoh:
  python scripts/06_evaluate_all.py              # run penuh (±1–3 menit di CPU)
  python scripts/06_evaluate_all.py --skip-cnn   # lewati inferensi CNN pada data B (lebih cepat)
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
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import src.analysis as an  # noqa: E402
import src.evaluate as ev  # noqa: E402
import src.plots as pl  # noqa: E402
from src.config import load_config, set_seed  # noqa: E402
from src.data_utils import load_metadata  # noqa: E402

ALGO_ML = ["logreg", "svm", "knn", "rf", "xgb"]


def setup_logger(path: Path) -> logging.Logger:
    log = logging.getLogger("tahap6")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def write_md(df: pd.DataFrame, name: str, header: str = ""):
    out = load_config().paths.tables / f"{name}.md"
    out.write_text((header + "\n\n" if header else "") + an.to_markdown(df), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# a. Tabel final
# ---------------------------------------------------------------------------
def step_a(log) -> dict:
    out = {}
    for X in ["A", "B"]:
        r = an.final_table(X)
        cols = ["model", "n", "n_pos", "prevalensi", "accuracy", "balanced_accuracy", "precision",
                "recall_sensitivity", "specificity", "f1", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
                "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "threshold", "val_auc", "cnn_seed_auc_mean",
                "cnn_seed_auc_std", "terbaik_val", "terbaik_test"]
        pl.save_table(r[[c for c in cols if c in r]], f"06_tabel_final_{X}", index=False)
        f = an.fmt_final_table(r)
        if X == "B":
            f = f.drop(columns=[c for c in f.columns if c.startswith("AUC test 3 seed")])
        pl.save_table(f, f"06_tabel_final_{X}_format", index=False)
        n, npos = int(r["n"].iloc[0]), int(r["n_pos"].iloc[0])
        write_md(f, f"06_tabel_final_{X}",
                 f"**Eksperimen {X}** — split test (n = {n}, COVID-19 = {npos}, prevalensi = {npos / n:.3f}). "
                 "Ambang = Youden's J yang dipilih di val. ★ = AUC val tertinggi (kriteria seleksi), "
                 "▲ = AUC test tertinggi (informasi).")
        bv = r.loc[r.terbaik_val, "model"].iloc[0]
        bt = r.loc[r.terbaik_test, "model"].iloc[0]
        log.info(f"[a] {X}: terbaik val = {bv}; AUC test tertinggi = {bt} "
                 f"({r.loc[r.terbaik_test, 'roc_auc'].iloc[0]:.3f})")
        out[X] = dict(terbaik_val=bv, terbaik_test=bt, tabel=r)
    return out


# ---------------------------------------------------------------------------
# b. Uji signifikansi
# ---------------------------------------------------------------------------
def step_b(final: dict, log) -> pd.DataFrame:
    rows = []
    for X in ["A", "B"]:
        s = an.load_scores(X)
        te = s[s.split == "test"]
        r = final[X]["tabel"]
        r = r[r.model != "Dummy (stratified)"].sort_values("roc_auc", ascending=False)
        top2 = r["model"].iloc[:2].tolist()
        pairs = [(top2[0], top2[1], "dua teratas (AUC test)")]
        if X == "A":
            pairs += [("SVM (RBF)", "CNN (log-mel)", "ML terbaik (val) vs CNN log-mel"),
                      ("SVM (RBF)", "CNN (MFCC 2D)", "ML terbaik (val) vs CNN MFCC 2D"),
                      ("SVM (RBF)", "Dummy (stratified)", "ML terbaik vs baseline")]
        else:
            pairs += [("SVM (RBF)", "Dummy (stratified)", "ML terbaik vs baseline")]
        for a, b, ket in pairs:
            d = ev.delong_test(te["label"], te[a], te[b])
            bs = ev.paired_bootstrap_auc_diff(te["label"], te[a], te[b], n_boot=2000)
            rows.append(dict(eksperimen=X, perbandingan=ket, model_1=a, model_2=b, auc_1=d["auc_a"],
                             auc_2=d["auc_b"], selisih=d["diff"], delong_se=d["se"], delong_z=d["z"],
                             delong_p=d["p"], delong_ci_low=d["ci_low"], delong_ci_high=d["ci_high"],
                             boot_ci_low=bs["ci_low"], boot_ci_high=bs["ci_high"], boot_p=bs["p"]))
            log.info(f"[b] {X} {a} − {b}: Δ={d['diff']:+.3f}  DeLong p={d['p']:.3f}  bootstrap p={bs['p']:.3f}")
    df = pd.DataFrame(rows)
    pl.save_table(df, "06_uji_signifikansi_auc", index=False)
    show = df[["eksperimen", "perbandingan", "model_1", "model_2", "auc_1", "auc_2", "selisih"]].copy()
    show["DeLong 95% CI"] = [f"[{lo:+.3f}; {hi:+.3f}]" for lo, hi in zip(df.delong_ci_low, df.delong_ci_high)]
    show["p DeLong"] = df["delong_p"]
    show["p bootstrap"] = df["boot_p"]
    write_md(show, "06_uji_signifikansi_auc",
             "Uji selisih ROC-AUC pada split test yang sama (DeLong; bootstrap berpasangan 2.000×).")
    return df


# ---------------------------------------------------------------------------
# c. ROC gabungan
# ---------------------------------------------------------------------------
def step_c(log):
    s = an.load_scores("A")
    te = s[s.split == "test"]
    names = ["SVM (RBF)", "CNN (log-mel)", "CNN (MFCC 2D)", "Dummy (stratified)"]
    fig = pl.plot_roc_pr_all(te["label"].values, {n: te[n].values for n in names},
                             title="Eksperimen A (test, n = 1.624): ML terbaik vs CNN vs baseline")
    pl.save_fig(fig, "06_roc_gabungan_A")
    plt.close(fig)
    log.info("[c] 06_roc_gabungan_A.png")


# ---------------------------------------------------------------------------
# d. Eksperimen B vs A & uji silang label
# ---------------------------------------------------------------------------
def _auc_row(set_name, label_name, src, model, y, s, n_boot=1000):
    y = np.asarray(y).astype(int)
    lo, hi = ev.bootstrap_ci(y, s, roc_auc_score, n_boot)
    return dict(set_evaluasi=set_name, label_evaluasi=label_name, dilatih_dengan=src, model=model,
                n=len(y), n_pos=int(y.sum()), roc_auc=roc_auc_score(y, s), ci_low=lo, ci_high=hi)


def step_d(args, log) -> dict:
    cfg = load_config()
    seen_A, seen_B = an.seen_uuids("A"), an.seen_uuids("B")
    tabA = pd.read_parquet(cfg.paths.features / "tabular_A.parquet")
    tabB = pd.read_parquet(cfg.paths.features / "tabular_B.parquet")
    manB = pd.read_csv(cfg.paths.processed / "manifest_B.csv")
    tabB = tabB.merge(manB[["uuid", "status_self_reported"]], on="uuid", how="left")

    # --- kesepakatan label self-reported vs ahli (seluruh manifest B)
    ct = pd.crosstab(manB["status_self_reported"].fillna("tidak diisi"), manB["label_name"], margins=True,
                     margins_name="Total")
    pl.save_table(ct, "06_kesepakatan_label_B")
    both = manB[manB.status_self_reported.isin(["COVID-19", "healthy"])]
    y_sr = (both.status_self_reported == "COVID-19").astype(int)
    kappa = cohen_kappa_score(y_sr, both["label"])
    agree = float((y_sr.values == both["label"].values).mean())
    ppv_sr = float(both.loc[y_sr == 1, "label"].mean())
    log.info(f"[d] kesepakatan label (n={len(both)}): setuju {agree:.3f}, kappa {kappa:.3f}; "
             f"dari yang mengaku COVID-19, {ppv_sr:.1%} juga dinilai COVID-19 oleh ahli")
    write_md(ct.reset_index().rename(columns={"status_self_reported": "status (self-reported) \\ diagnosis ahli"}),
             "06_kesepakatan_label_B",
             f"Rekaman Eksperimen B (n = {len(manB)}): status self-reported vs diagnosis ahli (majority vote). "
             f"Pada {len(both)} rekaman yang keduanya biner: kesepakatan = {agree:.3f}, Cohen's κ = {kappa:.3f}.")

    rows = []
    # S1: test B tanpa uuid train/val A → model A vs model B, label ahli
    s1 = tabB[(tabB.split == "test") & ~tabB.uuid.isin(seen_A)]
    # S2: test A tanpa uuid train/val B → model A vs model B, label self-reported
    s2 = tabA[(tabA.split == "test") & ~tabA.uuid.isin(seen_B)]
    # S3: semua rekaman B yang tidak dilihat model A DAN punya status biner → label ahli vs self-reported
    s3 = tabB[~tabB.uuid.isin(seen_A) & tabB.status_self_reported.isin(["COVID-19", "healthy"])]
    s3_ysr = (s3.status_self_reported == "COVID-19").astype(int)
    n1, n2, n3 = "S1: test B (label ahli)", "S2: test A (label self-reported)", "S3: rekaman B tak terlihat model A"
    log.info(f"[d] S1 n={len(s1)} (pos {int(s1.label.sum())}), S2 n={len(s2)} (pos {int(s2.label.sum())}), "
             f"S3 n={len(s3)} (ahli pos {int(s3.label.sum())}, self-reported pos {int(s3_ysr.sum())})")

    for algo in tqdm(ALGO_ML, desc="uji silang ML"):
        name = an.NAMA_MODEL[algo]
        rows.append(_auc_row(n1, "ahli", "label self-reported (A)", name, s1.label, an.ml_predict(algo, "A", s1)))
        rows.append(_auc_row(n1, "ahli", "label ahli (B)", name, s1.label, an.ml_predict(algo, "B", s1)))
        rows.append(_auc_row(n2, "self-reported", "label self-reported (A)", name, s2.label,
                             an.ml_predict(algo, "A", s2)))
        rows.append(_auc_row(n2, "self-reported", "label ahli (B)", name, s2.label, an.ml_predict(algo, "B", s2)))
        p3 = an.ml_predict(algo, "A", s3)
        rows.append(_auc_row(n3 + " — label ahli", "ahli", "label self-reported (A)", name, s3.label, p3))
        rows.append(_auc_row(n3 + " — label self-reported", "self-reported", "label self-reported (A)", name,
                             s3_ysr, p3))

    if not args.skip_cnn:
        for ck, name in [("cnn_logmel_A.pt", "CNN (log-mel)"), ("cnn_mfcc2d_A.pt", "CNN (MFCC 2D)")]:
            t0 = time.time()
            p = an.cnn_predict(ck, "B", set(s1.uuid) | set(s3.uuid))
            ps = p.set_index("uuid")["skor"]
            rows.append(_auc_row(n1, "ahli", "label self-reported (A)", name, s1.label, ps.loc[s1.uuid].values))
            rows.append(_auc_row(n3 + " — label ahli", "ahli", "label self-reported (A)", name, s3.label,
                                 ps.loc[s3.uuid].values))
            rows.append(_auc_row(n3 + " — label self-reported", "self-reported", "label self-reported (A)", name,
                                 s3_ysr, ps.loc[s3.uuid].values))
            log.info(f"[d] inferensi {name} pada data B: {time.time() - t0:.0f} s")

    df = pd.DataFrame(rows)
    pl.save_table(df, "06_uji_silang_label", index=False)
    piv = df.assign(auc=[f"{a:.3f} [{lo:.3f}–{hi:.3f}]" for a, lo, hi in zip(df.roc_auc, df.ci_low, df.ci_high)])
    piv = piv.pivot_table(index="model", columns=["set_evaluasi", "dilatih_dengan"], values="auc", aggfunc="first")
    piv = piv.reindex([m for m in an.URUTAN_MODEL if m in piv.index])
    piv.columns = [f"{a} ← {b}" for a, b in piv.columns]
    write_md(piv.reset_index().fillna("–"), "06_uji_silang_label",
             f"ROC-AUC [CI 95%] uji silang. Kolom = set evaluasi ← sumber label training. "
             f"S1 n = {len(s1)} ({int(s1.label.sum())} COVID-19 ahli); S2 n = {len(s2)} ({int(s2.label.sum())} "
             f"COVID-19 self-reported); S3 n = {len(s3)} (prediksi identik, dua label). "
             "uuid train/val model sumber dibuang dari set evaluasi.")

    fig = pl.plot_cross_label(df[df.model == "SVM (RBF)"].assign(
        set_evaluasi=lambda d: d.set_evaluasi.str.replace(" — ", "\n").str.replace(": ", ":\n")),
        title="Uji silang sumber label (SVM RBF, ROC-AUC [CI 95%])")
    pl.save_fig(fig, "06_uji_silang_label")
    plt.close(fig)
    return dict(kappa=float(kappa), kesepakatan=agree, n_kesepakatan=int(len(both)), ppv_self_reported=ppv_sr,
                n_s1=len(s1), n_s2=len(s2), n_s3=len(s3),
                svm=df[df.model == "SVM (RBF)"][["set_evaluasi", "dilatih_dengan", "roc_auc"]].to_dict("records"))


# ---------------------------------------------------------------------------
# e. Analisis error & subgrup
# ---------------------------------------------------------------------------
def step_e(log) -> dict:
    cfg = load_config()
    thr = an.thresholds_from_results("A")
    s = an.load_scores("A")
    te = s[s.split == "test"].reset_index(drop=True)
    names = ["SVM (RBF)", "CNN (log-mel)", "CNN (MFCC 2D)"]

    # e1. distribusi skor per kelas
    fig = pl.plot_score_distributions(te["label"].values, {n: te[n].values for n in names}, thr,
                                      title="Distribusi skor prediksi per kelas — Eksperimen A (test)")
    pl.save_fig(fig, "06_distribusi_skor_A")
    plt.close(fig)
    dist = []
    for n in names:
        for k, lab in [(0, "healthy"), (1, "COVID-19")]:
            v = te.loc[te.label == k, n]
            dist.append(dict(model=n, kelas=lab, n=len(v), median=v.median(), p05=v.quantile(.05),
                             p95=v.quantile(.95), ambang=thr[n], persen_di_atas_ambang=float((v >= thr[n]).mean())))
    pl.save_table(pd.DataFrame(dist), "06_distribusi_skor_A", index=False)

    # e2. contoh FP/FN paling yakin (SVM, model ML terbaik menurut val)
    meta = load_metadata(add_paths=False)
    man = pd.read_csv(cfg.paths.processed / "manifest_A.csv")
    te = an.add_subgroups(te, meta).merge(man[["uuid", "durasi_trim_s", "cough_detected", "n_segmen"]],
                                          on="uuid", how="left")
    m = "SVM (RBF)"
    pred = (te[m] >= thr[m]).astype(int)
    te["jenis"] = np.select([(pred == 1) & (te.label == 1), (pred == 0) & (te.label == 1),
                             (pred == 1) & (te.label == 0)], ["TP", "FN", "FP"], "TN")
    fp = te[te.jenis == "FP"].nlargest(3, m)
    fn = te[te.jenis == "FN"].nsmallest(3, m)
    X = np.load(cfg.paths.features / "logmel_A_test.npy", mmap_mode="r")
    u = np.load(cfg.paths.features / "uuids_A_test.npy", allow_pickle=True)
    pos = {k: i for i, k in enumerate(u)}
    from src.models_cnn import content_length
    items, contoh = [], []
    for jenis, d in [("FP", fp), ("FN", fn)]:
        for _, r in d.iterrows():
            spec = np.asarray(X[pos[r.uuid]], dtype=np.float32)
            spec = spec[:, :max(content_length(spec), 50)]
            umur = "?" if pd.isna(r.get("kelompok_umur")) else r.kelompok_umur
            items.append(dict(spec=spec, judul=(
                f"{jenis} — skor {r[m]:.2f} (ambang {thr[m]:.2f})\nlabel {'COVID-19' if r.label else 'healthy'}, "
                f"{r.gender}, umur {umur}, kond. napas {r.respiratory_condition}\n"
                f"uuid {r.uuid[:8]}…, durasi {r.durasi_trim_s:.1f} s")))
            contoh.append(dict(jenis=jenis, uuid=r.uuid, skor_svm=r[m], label=int(r.label),
                               skor_cnn_logmel=r["CNN (log-mel)"], skor_cnn_mfcc=r["CNN (MFCC 2D)"],
                               gender=r.gender, kelompok_umur=umur, respiratory_condition=r.respiratory_condition,
                               fever_muscle_pain=r.fever_muscle_pain, durasi_trim_s=r.durasi_trim_s,
                               cough_detected=r.cough_detected))
    fig = pl.plot_error_examples(items)
    fig.suptitle("Contoh kesalahan paling yakin SVM (RBF) — baris atas: false positive, bawah: false negative",
                 y=1.02)
    pl.save_fig(fig, "06_contoh_fp_fn_A")
    plt.close(fig)
    pl.save_table(pd.DataFrame(contoh), "06_contoh_fp_fn_A", index=False)

    # e3. karakteristik per jenis keputusan
    kar = te.groupby("jenis").agg(
        n=("uuid", "size"), skor_median=(m, "median"), durasi_trim_median_s=("durasi_trim_s", "median"),
        cough_detected_median=("cough_detected", "median"), tanpa_segmen_persen=("n_segmen", lambda v: (v == 0).mean()),
        kondisi_napas_ya_persen=("respiratory_condition", lambda v: (v == "ya").mean()),
        demam_ya_persen=("fever_muscle_pain", lambda v: (v == "ya").mean()),
        laki_laki_persen=("gender", lambda v: (v == "male").mean()),
    ).reindex(["TP", "FN", "FP", "TN"])
    pl.save_table(kar, "06_karakteristik_error_A")
    write_md(kar.reset_index(), "06_karakteristik_error_A",
             f"Karakteristik rekaman per jenis keputusan SVM (RBF), Eksperimen A test, ambang {thr[m]:.3f}.")
    log.info("[e] karakteristik error:\n" + kar.round(3).to_string())
    from scipy.stats import spearmanr
    kor = pd.DataFrame([dict(model=n, kelas=k, rho_spearman_skor_vs_durasi=spearmanr(d[n], d.durasi_trim_s)[0],
                             auc_durasi_saja=roc_auc_score(te.label, te.durasi_trim_s))
                        for n in names for k, d in [("semua", te), ("healthy", te[te.label == 0])]])
    pl.save_table(kor, "06_korelasi_skor_durasi_A", index=False)
    log.info("[e] korelasi skor vs durasi:\n" + kor.round(3).to_string())

    # e4. subgrup (fairness)
    rows, overall = [], {}
    for n in names:
        overall[n] = roc_auc_score(te.label, te[n])
        for col, lab in an.KOLOM_SUBGRUP.items():
            sg = ev.subgroup_metrics(te.label, te[n], te[col], thr[n], n_boot=500)
            sg.insert(0, "variabel", lab)
            sg.insert(0, "model", n)
            rows.append(sg)
    # Eksperimen B (SVM) sebagai pembanding
    sB = an.load_scores("B")
    teB = an.add_subgroups(sB[sB.split == "test"], meta)
    thrB = an.thresholds_from_results("B")
    for col, lab in an.KOLOM_SUBGRUP.items():
        sg = ev.subgroup_metrics(teB.label, teB["SVM (RBF)"], teB[col], thrB["SVM (RBF)"], n_boot=500)
        sg.insert(0, "variabel", lab)
        sg.insert(0, "model", "SVM (RBF) — B")
        rows.append(sg)
    sub = pd.concat(rows, ignore_index=True)
    order = {k: i for i, k in enumerate(["female", "male", "other", "< 25", "25–34", "35–44", "45–54", "≥ 55",
                                         "tidak", "ya", "tidak diisi"])}
    sub["_o"] = sub["subgrup"].map(order).fillna(99)
    sub = sub.sort_values(["model", "variabel", "_o"]).drop(columns="_o").reset_index(drop=True)
    pl.save_table(sub, "06_subgrup_fairness", index=False)
    sub.attrs["overall"] = overall
    for n, fname in [("SVM (RBF)", "06_subgrup_svm_A"), ("CNN (log-mel)", "06_subgrup_cnn_logmel_A")]:
        fig = pl.plot_subgroup_forest(sub, n, title=f"AUC per subgrup — {n}, Eksperimen A (test)")
        pl.save_fig(fig, fname)
        plt.close(fig)
    # Markdown: SVM A detail + AUC model lain
    a = sub[sub.model == "SVM (RBF)"].copy()
    a["AUC SVM [CI 95%]"] = [("–" if pd.isna(v) else f"{v:.3f} [{lo:.3f}–{hi:.3f}]")
                             for v, lo, hi in zip(a.roc_auc, a.roc_auc_ci_low, a.roc_auc_ci_high)]
    for n in ["CNN (log-mel)", "CNN (MFCC 2D)"]:
        b = sub[sub.model == n].set_index(["variabel", "subgrup"])["roc_auc"]
        a[f"AUC {n}"] = [b.get((v, s_), np.nan) for v, s_ in zip(a.variabel, a.subgrup)]
    md = a[["variabel", "subgrup", "n", "n_pos", "prevalensi", "AUC SVM [CI 95%]", "recall_sensitivity",
            "specificity", "AUC CNN (log-mel)", "AUC CNN (MFCC 2D)"]].rename(
        columns={"recall_sensitivity": "Sens. SVM", "specificity": "Spes. SVM"})
    write_md(md, "06_subgrup_fairness",
             "Performa per subgrup, Eksperimen A test (ambang Youden dari val). AUC hanya bila subgrup punya "
             "≥ 5 positif dan ≥ 5 negatif. Keseluruhan: " + ", ".join(f"{k} {v:.3f}" for k, v in overall.items()))
    gap = (sub[sub.roc_auc.notna()].groupby(["model", "variabel"])["roc_auc"].agg(["min", "max"])
           .assign(selisih=lambda d: d["max"] - d["min"]))
    pl.save_table(gap, "06_subgrup_selisih_auc")
    log.info("[e] selisih AUC antar-subgrup:\n" + gap.round(3).to_string())
    return dict(karakteristik=kar.round(4).reset_index().to_dict("records"),
                korelasi_durasi=kor.round(4).to_dict("records"),
                selisih_subgrup=gap.round(4).reset_index().to_dict("records"))


# ---------------------------------------------------------------------------
# f. Perbandingan dengan literatur
# ---------------------------------------------------------------------------
LITERATUR = [
    dict(referensi="R0 — Orlandic dkk. (2021), Sci. Data 8:156",
         dataset="COUGHVID (sumber dataset); >20.000 rekaman batuk crowdsourced (angka v1, dikutip R1 hlm. 145)",
         label="self-reported + label ahli (4 pulmonolog) untuk sebagian rekaman",
         fitur_model="Detektor batuk (XGBoost) untuk `cough_detected`; bukan klasifikator COVID-19 utama",
         protokol="lihat paper",
         metrik="lihat paper [TODO: cek manual — PDF R0 tidak ada di references/]",
         catatan="Paper dataset; bukan benchmark diagnosis. Angka spesifik belum diverifikasi."),
    dict(referensi="R2 — Lella & Pja (2022), Alexandria Eng. J. 61:1319–1334",
         dataset="Cambridge 'COVID-19 Sounds' (batuk, napas, suara); ±1.539 track (hlm. 1328), ±300 pengguna "
                 "positif (hlm. 1322) — BUKAN COUGHVID",
         label="status dari aplikasi (kemungkinan self-reported) [TODO: cek manual]",
         fitur_model="De-noising autoencoder + GFCC + IMFCC (multi-feature channel) → DCNN, softmax",
         protokol="5 folder / k-fold, tidak per subjek; augmentasi (time stretch, pitch shift, DRC, noise; MUDA) — "
                  "urutan augmentasi vs split tidak dinyatakan",
         metrik="Accuracy 95,45% / F1 96,96% (napas+batuk+suara, Tabel 2 hlm. 1328); batuk saja accuracy 93,12% / "
                "F1 94,13% (Tabel 4 hlm. 1332) [TODO: cek manual — label tugas di Tabel 2–5 tidak konsisten]. "
                "Tidak melaporkan AUC",
         catatan="Dataset, modalitas, dan metrik berbeda; augmentasi berpotensi sebelum split (risiko leakage)."),
    dict(referensi="R3 — Kranthi Kumar & Alphonse (2022), Eur. Phys. J. ST 231:3329–3346",
         dataset="Cambridge 'COVID-19 Sounds' (batuk, napas, suara); 1.541 file (hlm. 3336) vs ±13.000 sampel "
                 "(hlm. 3338) [TODO: cek manual] — BUKAN COUGHVID",
         label="tidak dinyatakan (mewarisi label aplikasi)",
         fitur_model="MFCC / E-GFCC / M-MFCC → CNN ringan (receptive field 1×12, 5 layer, BN)",
         protokol="5 folder; pembagian 10%/10%/80% (kalimat hlm. 3338 kontradiktif); tidak per subjek; "
                  "augmentasi disebut, urutan terhadap split tidak dinyatakan",
         metrik="Accuracy 91,32% / F1 92,48% (M-MFCC, 1×12; Tabel 1 hlm. 3341); batuk+napas+suara accuracy 92,32% "
                "/ F1 93,48% (Tabel 3 hlm. 3342). Tidak melaporkan AUC",
         catatan="Dataset/modalitas berbeda; hanya accuracy & F1 (bias pada data seimbang); jumlah data tidak jelas."),
    dict(referensi="R4 — Xiong dkk. (2022), Sci. Rep. 12:21990",
         dataset="Rekaman batuk call-centre MedInGroup: 698 rekaman (1 per subjek; hlm. 3) — BUKAN COUGHVID",
         label="PCR-confirmed (terverifikasi) vs patient-reported (tidak terverifikasi); non-COVID self-reported",
         fitur_model="openSMILE 6.373 fitur + seleksi ElasticNet → LR, LDA, RF, XGB, DNN, CNN-RNN, CA-CNN",
         protokol="per subjek; train 274 subjek, test 63 subjek berlabel PCR (20 COVID / 43 non); diulang 5×; "
                  "tanpa augmentasi",
         metrik="RF AUC 0,83 (label PCR) vs 0,79 (label self-reported); CA-CNN AUC 0,80 vs 0,47 (Tabel 2 hlm. 5) "
                "[TODO: cek manual — teks hlm. 4 menyebut selisih 0,02–0,10]",
         catatan="Setting paling mirip dengan kita (AUC, data tak seimbang, isu label); test berlabel PCR, kita "
                 "tidak punya PCR."),
]


def step_f(final: dict, log):
    rA, rB = final["A"]["tabel"].set_index("model"), final["B"]["tabel"].set_index("model")
    sv = rA.loc["SVM (RBF)"]
    kami = dict(
        referensi="Penelitian ini",
        dataset="COUGHVID v3, batuk saja, cough_detected ≥ 0,8. A: 10.852 rekaman (720 COVID-19); "
                "B: 1.013 rekaman berlabel ahli (470 COVID-19)",
        label="A: self-reported; B: majority vote diagnosis pulmonolog",
        fitur_model="315 fitur tabular (MFCC±Δ, GFCC, spektral, pitch, RMS/ZCR) → LR/SVM/k-NN/RF/XGBoost; "
                    "log-mel & MFCC 2D → LightCNN (97 ribu parameter)",
        protokol="split per uuid 70/15/15 SEBELUM undersampling/SMOTE/augmentasi/scaling; tuning di train, "
                 "ambang di val, test sekali; tanpa ID subjek",
        metrik=(f"A: AUC test SVM {sv.roc_auc:.3f} [{sv.roc_auc_ci_low:.3f}–{sv.roc_auc_ci_high:.3f}], "
                f"CNN log-mel {rA.loc['CNN (log-mel)', 'roc_auc']:.3f}, CNN MFCC 2D "
                f"{rA.loc['CNN (MFCC 2D)', 'roc_auc']:.3f} (3 seed: {rA.loc['CNN (MFCC 2D)', 'cnn_seed_auc_mean']:.3f}"
                f" ± {rA.loc['CNN (MFCC 2D)', 'cnn_seed_auc_std']:.3f}); PR-AUC ≈ 0,09 (prevalensi 0,067); "
                f"accuracy SVM {sv.accuracy:.3f}. B: AUC test SVM {rB.loc['SVM (RBF)', 'roc_auc']:.3f} "
                f"[{rB.loc['SVM (RBF)', 'roc_auc_ci_low']:.3f}–{rB.loc['SVM (RBF)', 'roc_auc_ci_high']:.3f}]"),
        catatan="Test dengan prevalensi alami (1:14); accuracy tidak informatif → AUC/PR-AUC utama.")
    df = pd.DataFrame(LITERATUR + [kami])
    df.columns = ["Referensi", "Dataset & data", "Label", "Fitur & model", "Protokol split / augmentasi",
                  "Metrik dilaporkan", "Catatan perbedaan setting"]
    pl.save_table(df, "06_perbandingan_literatur", index=False)
    write_md(df, "06_perbandingan_literatur",
             "Perbandingan dengan literatur. Angka literatur dikutip dari PDF di `references/` (nomor halaman "
             "jurnal); **[TODO: cek manual]** = angka/klaim yang ambigu di sumber atau belum diverifikasi. "
             "R2–R4 tidak memakai COUGHVID → angka **tidak dapat dibandingkan langsung**.")
    log.info("[f] 06_perbandingan_literatur.csv/md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-cnn", action="store_true", help="lewati inferensi CNN pada data B (uji silang)")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg.seed)
    pl.setup_style()
    log = setup_logger(cfg.paths.logs / "evaluate_all.log")
    t0 = time.time()
    final = step_a(log)
    sig = step_b(final, log)
    step_c(log)
    d = step_d(args, log)
    e = step_e(log)
    step_f(final, log)
    summary = dict(
        terbaik={X: {k: v for k, v in final[X].items() if k != "tabel"} for X in final},
        signifikansi=sig.round(4).to_dict("records"), uji_silang=d, error=e,
        waktu_s=round(time.time() - t0, 1))
    (cfg.paths.logs / "evaluate_all_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log.info(f"Selesai dalam {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
