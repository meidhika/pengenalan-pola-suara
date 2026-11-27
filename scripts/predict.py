"""Demo inferensi: satu (atau beberapa) file audio batuk → probabilitas & label COVID-19.

!!! BUKAN ALAT DIAGNOSIS MEDIS !!!
Model ini dibuat untuk tugas kuliah. Pada data uji, ROC-AUC model Eksperimen A (label self-reported)
hanya ±0,58 dengan precision ±0,08 — hasilnya tidak boleh dipakai untuk keputusan kesehatan apa pun.
Gunakan tes PCR/antigen dan konsultasikan dengan tenaga kesehatan.

Langkah yang dijalankan SAMA dengan pipeline training (src/preprocessing.py & src/features.py):
  a. ffmpeg → WAV mono 16 kHz PCM 16-bit   b. trim senyap (top_db 30)   c. segmentasi batuk (info)
  d. reduksi noise (spectral gating)       e. normalisasi peak 0,99 → simpan sementara 16-bit
  → 315 fitur tabular (NaN f0 → 0) → pipeline model tersimpan (scaler + SelectKBest + klasifikator)
  → probabilitas P(COVID-19) dan label dengan ambang Youden yang dipilih di data validasi.
File sementara ditulis ke folder temp sistem, bukan ke data/.

Contoh:
  python scripts/predict.py "rekaman batuk.webm"
  python scripts/predict.py a.wav b.ogg --eksperimen B          # model berlabel ahli
  python scripts/predict.py a.mp3 --algo rf --json
  python scripts/predict.py a.wav --semua                       # tampilkan semua model ML
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import features as fx  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.config import load_config  # noqa: E402

DISCLAIMER = ("PERINGATAN: ini demo tugas kuliah, BUKAN alat diagnosis medis. ROC-AUC test hanya ±0,58 "
              "(model A, label self-reported); model B (0,83) diukur terhadap penilaian pulmonolog, bukan PCR. "
              "Hasil tidak dapat dipakai untuk menyimpulkan seseorang positif/negatif COVID-19. "
              "Gunakan tes PCR/antigen dan konsultasikan dengan tenaga kesehatan.")
NAMA = {"svm": "SVM (RBF)", "logreg": "Logistic Regression", "knn": "k-NN", "rf": "Random Forest",
        "xgb": "XGBoost"}
EKSP = {"A": "dilatih dengan label self-reported", "B": "dilatih dengan label diagnosis pulmonolog"}
FORMAT_DIDUKUNG = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac", ".opus"}


def preprocess_audio(src: Path, cfg) -> tuple[np.ndarray, int, dict]:
    """Langkah a–e Tahap 2 untuk satu file di luar dataset. Return (sinyal final, sr, info)."""
    P, A = cfg.preprocessing, cfg.audio
    sr = A.sample_rate
    with tempfile.TemporaryDirectory(prefix="ppkb_predict_") as td:
        td = Path(td)
        conv = pp.convert_to_wav(src, td / "konversi.wav", sr=sr, bit_depth=P.bit_depth, overwrite=True)
        y0, sr0 = pp.load_wav(conv)
        if sr0 != sr or y0.size == 0 or not np.isfinite(y0).all():
            raise pp.PreprocessError("audio kosong / tidak valid")
        if float(np.max(np.abs(y0))) < 1e-4:
            raise pp.PreprocessError("audio senyap (amplitudo maks < 1e-4)")
        y1, _ = pp.trim_silence(y0, **P.trim)
        if len(y1) / sr < P.min_duration_s:
            raise pp.PreprocessError(f"terlalu pendek setelah trim ({len(y1) / sr:.2f} s < {P.min_duration_s} s)")
        segs = pp.segment_coughs(y1, sr, **P.segmentation)
        nrc = P.noise_reduction
        y2 = pp.reduce_noise(y1, sr, stationary=nrc.stationary, prop_decrease=nrc.prop_decrease) \
            if nrc.enabled else y1
        nm = P.normalization
        y3 = pp.normalize(y2, method=nm.method, peak_target=nm.peak_target, rms_target_db=nm.rms_target_db,
                          target_duration_s=None, sr=sr)
        # simpan & baca ulang agar kuantisasi 16-bit identik dengan wav yang dipakai saat training
        out = pp.save_wav(td / "final.wav", y3, sr, bit_depth=P.bit_depth)
        y, _ = pp.load_wav(out)
    info = {"durasi_asli_s": round(len(y0) / sr, 2), "durasi_setelah_trim_s": round(len(y1) / sr, 2),
            "n_segmen_batuk": len(segs)}
    if len(y0) / sr > A.max_duration_s + 0.5:
        info["catatan"] = f"rekaman > {A.max_duration_s:g} s (data training umumnya ≤ 10 s)"
    return y, sr, info


def tabular_features(y: np.ndarray, sr: int, cfg) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = fx.extract_tabular(y, sr, cfg)
    df = pd.DataFrame([feats])
    df, _ = fx.clean_tabular(df, list(feats))   # aturan NaN yang sama dengan Tahap 3
    return df


def load_model(algo: str, eksperimen: str, cfg) -> dict:
    path = cfg.paths.models / f"ml_{algo}_{eksperimen}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ada — jalankan dulu scripts/04_train_ml.py")
    return joblib.load(path)


def predict_one(path: Path, models: dict[str, dict], cfg) -> dict:
    y, sr, info = preprocess_audio(path, cfg)
    X = tabular_features(y, sr, cfg)
    hasil = []
    for algo, m in models.items():
        p = float(m["pipeline"].predict_proba(X[m["features"]])[:, 1][0])
        thr = float(m["threshold"])
        hasil.append({"model": NAMA.get(algo, algo), "eksperimen": m["eksperimen"], "prob_covid": round(p, 4),
                      "ambang": round(thr, 4), "label": "COVID-19" if p >= thr else "healthy"})
    return {"file": str(path), **info, "prediksi": hasil}


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo prediksi suara batuk COVID-19 (BUKAN alat diagnosis medis).")
    ap.add_argument("audio", nargs="+", type=Path, help="file audio (wav/webm/ogg/mp3/…)")
    ap.add_argument("--eksperimen", choices=["A", "B"], default="A",
                    help="A = model label self-reported (eksperimen utama, default); B = model label ahli")
    ap.add_argument("--algo", choices=list(NAMA), default="svm",
                    help="algoritma (default svm = ML terbaik menurut AUC validasi di A dan B)")
    ap.add_argument("--semua", action="store_true", help="tampilkan prediksi semua model ML eksperimen ini")
    ap.add_argument("--json", action="store_true", help="keluaran JSON")
    args = ap.parse_args()

    cfg = load_config()
    algos = list(NAMA) if args.semua else [args.algo]
    models = {a: load_model(a, args.eksperimen, cfg) for a in algos}

    semua, kode = [], 0
    for f in args.audio:
        if not f.exists():
            semua.append({"file": str(f), "error": "file tidak ditemukan"}); kode = 1; continue
        if f.suffix.lower() not in FORMAT_DIDUKUNG:
            print(f"[info] ekstensi {f.suffix} tidak umum; tetap dicoba lewat ffmpeg.", file=sys.stderr)
        try:
            semua.append(predict_one(f, models, cfg))
        except pp.PreprocessError as e:
            semua.append({"file": str(f), "error": f"audio ditolak: {e}"}); kode = 1

    if args.json:
        print(json.dumps({"disclaimer": DISCLAIMER, "hasil": semua}, ensure_ascii=False, indent=2))
        return kode

    print("=" * 78)
    print(DISCLAIMER)
    print("=" * 78)
    print(f"Model: Eksperimen {args.eksperimen} ({EKSP[args.eksperimen]})")
    for r in semua:
        print(f"\nFile : {r['file']}")
        if "error" in r:
            print(f"  GAGAL: {r['error']}")
            continue
        print(f"  durasi {r['durasi_asli_s']} s → {r['durasi_setelah_trim_s']} s setelah trim; "
              f"segmen batuk terdeteksi: {r['n_segmen_batuk']}")
        if r.get("catatan"):
            print(f"  catatan: {r['catatan']}")
        if r["n_segmen_batuk"] == 0:
            print("  catatan: tidak ada segmen batuk terdeteksi — pastikan rekaman berisi suara batuk.")
        for p in r["prediksi"]:
            print(f"  {p['model']:<20s} P(COVID-19) = {p['prob_covid']:.3f}  (ambang {p['ambang']:.3f})"
                  f"  → {p['label']}")
    print("\nProbabilitas tidak terkalibrasi (train di-undersample); bandingkan dengan ambang, bukan 0,5.")
    return kode


if __name__ == "__main__":
    sys.exit(main())
