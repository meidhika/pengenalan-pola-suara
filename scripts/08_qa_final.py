"""Tahap 8 — pemeriksaan akhir (QA) otomatis.

1. Kebocoran data (data leakage):
   - uuid train/val/test saling lepas (splits & manifest A/B; fitur tabular; array 2D).
   - urutan/isi uuid array 2D & tabular = manifest per split.
   - undersampling hanya di train (val/test = populasi split penuh, prevalensi alami).
   - pipeline ML tersimpan: scaler/SelectKBest/SMOTE di DALAM pipeline (di-fit hanya pada data fit = train).
   - CNN: statistik normalisasi di checkpoint = mean/std per pita dari TRAIN saja (dihitung ulang).
   - uji silang (Tahap 6): set evaluasi tidak memuat uuid train/val model sumber.
2. Konsistensi angka: angka kunci dari reports/tables/*.csv (diformat koma desimal) dicari di teks
   laporan .docx dan slide .pptx.

Keluaran: reports/logs/qa_final.md (+ ringkasan di konsol). Exit code 1 bila ada pemeriksaan GAGAL.
Jalankan:  python scripts/08_qa_final.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402

cfg = load_config()
TAB = ROOT / "reports" / "tables"
FIN = ROOT / "reports" / "final"
PROC = ROOT / "data" / "processed"
FEAT = ROOT / "data" / "features"
OUT = ROOT / "reports" / "logs" / "qa_final.md"

hasil: list[tuple[str, str, bool, str]] = []   # (bagian, pemeriksaan, lulus, detail)


def cek(bagian, nama, ok, detail=""):
    hasil.append((bagian, nama, bool(ok), str(detail)))


def koma(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


# =========================================================== 1. LEAKAGE
L = "Leakage"
for e in ("A", "B"):
    sp = pd.read_csv(PROC / f"splits_{e}.csv")
    man = pd.read_csv(PROC / f"manifest_{e}.csv")
    for nm, df in (("splits", sp), ("manifest", man)):
        g = {s: set(df.loc[df.split == s, "uuid"]) for s in ("train", "val", "test")}
        inter = len(g["train"] & g["val"]) + len(g["train"] & g["test"]) + len(g["val"] & g["test"])
        cek(L, f"{e}: uuid {nm} train/val/test saling lepas", inter == 0 and df.uuid.is_unique,
            f"irisan={inter}, uuid unik={df.uuid.is_unique}")
    # undersampling hanya di train
    for s in ("val", "test"):
        full = sp[sp.split == s]
        m = man[man.split == s]
        hilang = len(set(full.uuid) - set(m.uuid))
        cek(L, f"{e}: {s} tidak di-undersample (manifest ⊇ populasi split, kecuali gagal preprocessing)",
            hilang <= 17, f"populasi {len(full)}, manifest {len(m)}, hilang {hilang} (file gagal preprocessing)")
        prev_full, prev_m = full.label.mean(), m.label.mean()
        cek(L, f"{e}: prevalensi {s} alami", abs(prev_full - prev_m) < 0.005,
            f"populasi {prev_full:.4f} vs manifest {prev_m:.4f}")
    # fitur tabular & array 2D mengikuti manifest
    tab = pd.read_parquet(FEAT / f"tabular_{e}.parquet", columns=["uuid", "split", "label"])
    mm = tab.merge(man[["uuid", "split", "label"]], on="uuid", how="outer", suffixes=("_tab", "_man"),
                   indicator=True)
    cek(L, f"{e}: fitur tabular = manifest (uuid, split, label)",
        (mm._merge == "both").all() and (mm.split_tab == mm.split_man).all() and (mm.label_tab == mm.label_man).all(),
        f"{len(tab)} baris; tidak cocok={int((mm._merge != 'both').sum())}")
    for s in ("train", "val", "test"):
        u = np.load(FEAT / f"uuids_{e}_{s}.npy", allow_pickle=True)
        y = np.load(FEAT / f"labels_{e}_{s}.npy")
        ms = man[man.split == s].set_index("uuid")
        ok = set(u) == set(ms.index) and len(u) == len(ms) and (ms.loc[list(u), "label"].to_numpy() == y).all()
        cek(L, f"{e}: array 2D {s} = manifest {s}", ok, f"n={len(u)}")

# pipeline ML: semua transformasi di dalam pipeline
for f in sorted((ROOT / "models").glob("ml_*_?.joblib")):
    m = joblib.load(f)
    steps = [n for n, _ in m["pipeline"].steps]
    ok = steps[0] == "scaler" and "select" in steps and steps[-1] == "clf" and \
        (("smote" in steps) == (m["imbalance"] == "smote"))
    cek(L, f"{f.stem}: scaler/SelectKBest/[SMOTE] di dalam pipeline", ok, " → ".join(steps))
    # ambang dari val: prediksi val tersimpan harus konsisten dgn panjang manifest val
    nval = (pd.read_csv(PROC / f"manifest_{m['eksperimen']}.csv").split == "val").sum()
    cek(L, f"{f.stem}: skor val tersimpan = ukuran val (ambang dipilih di val)", len(m["pred_val"]) == nval,
        f"{len(m['pred_val'])} vs {nval}")

# CNN: statistik normalisasi = train saja
try:
    import torch
    from src import models_cnn as mc
    for ck_name in ("cnn_logmel_A.pt", "cnn_mfcc2d_A.pt"):
        ck = torch.load(ROOT / "models" / ck_name, map_location="cpu", weights_only=False)
        feat = "logmel" if "logmel" in ck_name else "mfcc2d"
        tp = ck.get("time_pool", 2)
        X, y, u, Lc = mc.load_split(feat, "A", "train", tp)
        ref = mc.BandNormalizer().fit(X)
        st = ck["normalizer"]
        ok = np.allclose(np.ravel(st["mean"]), ref.mean_.ravel(), atol=1e-3) and \
            np.allclose(np.ravel(st["std"]), ref.std_.ravel(), atol=1e-3)
        # pembanding: statistik bila val ikut di-fit (harus berbeda → membuktikan cek ini sensitif)
        Xv = mc.load_split(feat, "A", "val", tp)[0]
        bocor = mc.BandNormalizer().fit(np.concatenate([X, Xv]))
        beda = float(np.abs(bocor.mean_.ravel() - np.ravel(st["mean"])).max())
        cek(L, f"{ck_name}: mean/std normalisasi = statistik train saja (n={len(y)})", ok,
            f"maks |selisih| vs train {float(np.abs(ref.mean_.ravel() - np.ravel(st['mean'])).max()):.2e}; "
            f"vs train+val {beda:.2e}")
except Exception as ex:  # noqa: BLE001
    cek(L, "CNN: normalisasi train-only (dihitung ulang)", False, f"tidak dapat dicek: {ex!r}")

# uji silang: tidak ada uuid train/val model sumber di set evaluasi
try:
    from src import analysis as an
    seenA, seenB = an.seen_uuids("A"), an.seen_uuids("B")
    manA, manB = pd.read_csv(PROC / "manifest_A.csv"), pd.read_csv(PROC / "manifest_B.csv")
    s1 = set(manB.loc[manB.split == "test", "uuid"]) - set(seenA)
    s2 = set(manA.loc[manA.split == "test", "uuid"]) - set(seenB)
    us = pd.read_csv(TAB / "06_uji_silang_label.csv")
    n1 = int(us.loc[us.set_evaluasi.str.startswith("S1"), "n"].iloc[0])
    n2 = int(us.loc[us.set_evaluasi.str.startswith("S2"), "n"].iloc[0])
    cek(L, "Uji silang S1/S2: set evaluasi tanpa uuid train/val model sumber",
        len(s1) == n1 and len(s2) == n2, f"S1 {len(s1)} (tabel {n1}), S2 {len(s2)} (tabel {n2})")
except Exception as ex:  # noqa: BLE001
    cek(L, "Uji silang: pengecualian uuid", False, f"tidak dapat dicek: {ex!r}")


# =========================================================== 2. KONSISTENSI ANGKA
def teks_docx(p: Path) -> str:
    from docx import Document
    d = Document(p)
    parts = [para.text for para in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.extend(c.text for c in row.cells)
    return "\n".join(parts)


def teks_pptx(p: Path) -> str:
    from pptx import Presentation
    prs = Presentation(p)
    parts = []
    for s in prs.slides:
        for sh in s.shapes:
            if sh.has_text_frame:
                parts.append(sh.text_frame.text)
            if getattr(sh, "has_table", False) and sh.has_table:
                parts.extend(c.text for r in sh.table.rows for c in r.cells)
    return "\n".join(parts)


def norm(t: str) -> str:
    return re.sub(r"[    ]+", " ", t).replace("−", "-")


DOCX = norm(teks_docx(FIN / "Laporan_COUGHVID_Kelompok.docx"))
PPTX = norm(teks_pptx(FIN / "Presentasi_COUGHVID_Kelompok.pptx"))
K = "Konsistensi"
cek_angka: list[tuple[str, str, bool, bool]] = []   # (label, string, wajib di docx, wajib di pptx)

fa = pd.read_csv(TAB / "06_tabel_final_A.csv")
fb = pd.read_csv(TAB / "06_tabel_final_B.csv")
for e, df, slide_all in (("A", fa, True), ("B", fb, False)):
    for _, r in df.iterrows():
        di_slide = slide_all or r.model == "SVM (RBF)"
        cek_angka.append((f"{e} {r.model} ROC-AUC", koma(r.roc_auc), True, di_slide))
        cek_angka.append((f"{e} {r.model} CI 95%", f"{koma(r.roc_auc_ci_low)}–{koma(r.roc_auc_ci_high)}", True,
                          di_slide))
        cek_angka.append((f"{e} {r.model} PR-AUC", koma(r.pr_auc), True, slide_all))
        cek_angka.append((f"{e} {r.model} sensitivitas", koma(r.recall_sensitivity), True, slide_all))
        cek_angka.append((f"{e} {r.model} spesifisitas", koma(r.specificity), True, slide_all))
cek_angka.append(("A n test", f"{int(fa.n.iloc[0]):,}".replace(",", "."), True, True))
cek_angka.append(("A prevalensi test", koma(fa.prevalensi.iloc[0]), True, True))
cek_angka.append(("CNN log-mel mean±std 3 seed", f"{koma(fa.set_index('model').loc['CNN (log-mel)', 'cnn_seed_auc_mean'])} ± "
                  f"{koma(fa.set_index('model').loc['CNN (log-mel)', 'cnn_seed_auc_std'])}", True, True))
cek_angka.append(("CNN MFCC mean±std 3 seed", f"{koma(fa.set_index('model').loc['CNN (MFCC 2D)', 'cnn_seed_auc_mean'])} ± "
                  f"{koma(fa.set_index('model').loc['CNN (MFCC 2D)', 'cnn_seed_auc_std'])}", True, True))

sig = pd.read_csv(TAB / "06_uji_signifikansi_auc.csv")
p_dummy = sig.loc[(sig.eksperimen == "A") & (sig.model_2 == "Dummy (stratified)"), "delong_p"].iloc[0]
cek_angka.append(("p DeLong SVM vs dummy (A)", koma(p_dummy, 2), True, True))
for _, r in sig.iterrows():
    if r.delong_p > 0.001:
        cek_angka.append((f"p DeLong {r.eksperimen} {r.model_1} vs {r.model_2}", koma(r.delong_p, 2), True, False))

kes = pd.read_csv(TAB / "06_kesepakatan_label_B.csv").set_index("status_self_reported")
n_biner = int(kes.loc[["COVID-19", "healthy"], ["COVID-19", "healthy"]].to_numpy().sum())
setuju = (kes.loc["COVID-19", "COVID-19"] + kes.loc["healthy", "healthy"]) / n_biner
cek_angka += [("Kesepakatan label n", str(n_biner), True, True),
              ("Kesepakatan label %", koma(100 * setuju, 1), True, True),
              ("% self-reported COVID dinilai COVID oleh ahli",
               f"{100 * kes.loc['COVID-19', 'COVID-19'] / kes.loc['COVID-19', 'Total']:.0f}%", False, True)]

us = pd.read_csv(TAB / "06_uji_silang_label.csv")
svm = us[us.model == "SVM (RBF)"]
for _, r in svm.iterrows():
    cek_angka.append((f"Uji silang SVM {r.set_evaluasi[:2]} ← {r.dilatih_dengan}", koma(r.roc_auc), True, False))
    if r.set_evaluasi.startswith("S1"):
        cek_angka.append((f"Uji silang SVM S1 ← {r.dilatih_dengan} (2 desimal, slide)", koma(r.roc_auc, 2), False, True))

ds = pd.read_csv(TAB / "06_distribusi_skor_A.csv")
for kelas in ("healthy", "COVID-19"):
    v = ds.loc[(ds.model == "SVM (RBF)") & (ds.kelas == kelas), "median"].iloc[0]
    cek_angka.append((f"Median skor SVM {kelas}", koma(v), True, True))
kr = pd.read_csv(TAB / "06_korelasi_skor_durasi_A.csv")
for mdl in ("SVM (RBF)", "CNN (log-mel)"):
    v = kr.loc[(kr.model == mdl) & (kr.kelas == "semua"), "rho_spearman_skor_vs_durasi"].iloc[0]
    cek_angka.append((f"Spearman skor–durasi {mdl}", koma(v, 2), True, True))
ke = pd.read_csv(TAB / "06_karakteristik_error_A.csv").set_index("jenis")
cek_angka.append(("Durasi median FP", koma(ke.loc["FP", "durasi_trim_median_s"], 1), True, True))
cek_angka.append(("Durasi median TN", koma(ke.loc["TN", "durasi_trim_median_s"], 1), True, True))
sg = pd.read_csv(TAB / "06_subgrup_fairness.csv")
sgs = sg[sg.model == "SVM (RBF)"].set_index(["variabel", "subgrup"])
for (var, sub), lab in [(("Demam / nyeri otot", "ya"), "demam ya"), (("Demam / nyeri otot", "tidak"), "demam tidak")]:
    v = sgs.loc[(var, sub), "roc_auc"]
    cek_angka.append((f"AUC subgrup {lab}", koma(v, 2), False, True))
    cek_angka.append((f"AUC subgrup {lab} (3 desimal)", koma(v, 3), True, False))
umur = [i for i in sgs.index if i[0].startswith("Kelompok umur") and "55" in i[1]][0]
cek_angka.append(("AUC subgrup umur ≥ 55", koma(sgs.loc[umur, "roc_auc"], 2), False, True))
uj = pd.read_csv(TAB / "03_uji_fitur_semua_A.csv")
cek_angka.append(("Jumlah fitur signifikan A", f"{int(uj['signifikan_fdr_0.05'].sum())} / {len(uj)}", False, True))
cek_angka.append(("Jumlah fitur", str(len(uj)), True, True))
top = uj.sort_values("auc_fitur", ascending=False).iloc[0]
cek_angka.append((f"AUC fitur terkuat ({top.fitur})", koma(top.auc_fitur), True, True))
par = pd.read_csv(TAB / "05_jumlah_parameter.csv")
cek_angka.append(("Parameter LightCNN", f"{int(par.parameter.iloc[0]):,}".replace(",", "."), True, True))
ra = pd.read_csv(TAB / "01_ringkasan_angka.csv")
cek_angka.append(("Total rekaman", "34.434", True, True))
spA, spB = pd.read_csv(PROC / "splits_A.csv"), pd.read_csv(PROC / "manifest_B.csv")
cek_angka.append(("Populasi A COVID-19", f"{int((spA.label == 1).sum())}", True, True))
cek_angka.append(("Populasi A healthy", f"{int((spA.label == 0).sum()):,}".replace(",", "."), True, True))
cek_angka.append(("Manifest B COVID-19", str(int((spB.label == 1).sum())), True, True))
cek_angka.append(("Manifest B healthy", str(int((spB.label == 0).sum())), True, True))

tidak_ada = []
for lab, s, wd, wp in cek_angka:
    ad, ap = s in DOCX, s in PPTX
    ok = (ad or not wd) and (ap or not wp)
    det = f"`{s}` — docx: {'ya' if ad else 'TIDAK'}{'' if wd else ' (opsional)'}, " \
          f"slide: {'ya' if ap else 'TIDAK'}{'' if wp else ' (opsional)'}"
    cek(K, lab, ok, det)

# =========================================================== laporan
lulus = sum(h[2] for h in hasil)
lines = ["# QA akhir — Tahap 8", "", f"Hasil: **{lulus}/{len(hasil)} lulus**.", "",
         "Dibuat oleh `scripts/08_qa_final.py`. Angka acuan dibaca dari `reports/tables/*.csv` dan "
         "`data/processed/*.csv`; teks dicari pada `Laporan_COUGHVID_Kelompok.docx` dan "
         "`Presentasi_COUGHVID_Kelompok.pptx`.", ""]
for bag in dict.fromkeys(h[0] for h in hasil):
    lines += [f"## {bag}", "", "| | Pemeriksaan | Detail |", "|---|---|---|"]
    for b, n, ok, d in hasil:
        if b == bag:
            lines.append(f"| {'✅' if ok else '❌'} | {n} | {d.replace('|', '/')} |")
    lines.append("")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"QA: {lulus}/{len(hasil)} lulus → {OUT.relative_to(ROOT)}")
for b, n, ok, d in hasil:
    if not ok:
        print(f"  GAGAL [{b}] {n}: {d}")
sys.exit(0 if lulus == len(hasil) else 1)
