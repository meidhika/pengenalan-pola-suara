"""Inferensi untuk audio di luar dataset (dipakai dashboard Streamlit `app_streamlit.py`).

Langkah identik dengan pipeline training (lihat juga scripts/predict.py):
  a. ffmpeg → WAV mono 16 kHz PCM 16-bit    b. trim senyap (top_db 30)    c. segmentasi batuk (info)
  d. reduksi noise (spectral gating)        e. normalisasi peak 0,99 → simpan/baca ulang 16-bit
  ML klasik : 315 fitur tabular → pipeline tersimpan (scaler + SelectKBest + klasifikator)
  CNN       : log-mel 64 × 1001 (pad/potong 10 s) → rata-rata 2 frame → standardisasi per pita
              (statistik TRAIN di checkpoint) → LightCNN → sigmoid
Label = "COVID-19" bila skor ≥ ambang Youden yang dipilih di data validasi (bukan 0,5).

BUKAN ALAT DIAGNOSIS MEDIS.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import features as fx
from src import preprocessing as pp
from src.config import load_config

DISCLAIMER = ("Ini demo tugas kuliah, **bukan alat diagnosis medis**. ROC-AUC data uji hanya ±0,58 "
              "(Eksperimen A, label *self-reported*); model B (±0,83) diukur terhadap penilaian pulmonolog, "
              "bukan PCR. Hasil tidak dapat dipakai untuk menyimpulkan seseorang positif/negatif COVID-19. "
              "Gunakan tes PCR/antigen dan konsultasikan dengan tenaga kesehatan.")

NAMA_ML = {"svm": "SVM (RBF)", "logreg": "Logistic Regression", "knn": "k-NN", "rf": "Random Forest",
           "xgb": "XGBoost"}
NAMA_CNN = {"logmel": "CNN (log-mel)", "mfcc2d": "CNN (MFCC 2D)"}
EKSP = {"A": "label self-reported (COVID-19 vs healthy)", "B": "label diagnosis pulmonolog"}
FORMAT_AUDIO = ["wav", "webm", "ogg", "mp3", "m4a", "flac", "opus", "aac", "mp4"]


# ---------------------------------------------------------------------------
# Preprocessing (dengan sinyal antara untuk visualisasi)
# ---------------------------------------------------------------------------
def preprocess_stages(src: str | Path, cfg=None) -> dict:
    """Jalankan langkah a–e untuk satu file. Return dict berisi sinyal tiap tahap + info ringkas.

    Kunci: y_raw (setelah konversi), y_trim, y_final (input fitur), sr, segmen (detik, relatif y_trim), info.
    Melempar pp.PreprocessError bila audio senyap/terlalu pendek/tidak valid.
    """
    cfg = cfg or load_config()
    P, A = cfg.preprocessing, cfg.audio
    sr = A.sample_rate
    with tempfile.TemporaryDirectory(prefix="ppkb_app_") as td:
        td = Path(td)
        conv = pp.convert_to_wav(src, td / "konversi.wav", sr=sr, bit_depth=P.bit_depth, overwrite=True)
        y0, sr0 = pp.load_wav(conv)
        if sr0 != sr or y0.size == 0 or not np.isfinite(y0).all():
            raise pp.PreprocessError("audio kosong / tidak valid")
        if float(np.max(np.abs(y0))) < 1e-4:
            raise pp.PreprocessError("audio senyap (amplitudo maksimum < 1e-4)")
        y1, (i0, i1) = pp.trim_silence(y0, **P.trim)
        if len(y1) / sr < P.min_duration_s:
            raise pp.PreprocessError(f"terlalu pendek setelah trim senyap ({len(y1) / sr:.2f} s < "
                                     f"{P.min_duration_s} s)")
        segs = pp.segment_coughs(y1, sr, **P.segmentation)
        nrc = P.noise_reduction
        y2 = pp.reduce_noise(y1, sr, stationary=nrc.stationary, prop_decrease=nrc.prop_decrease) \
            if nrc.enabled else y1
        nm = P.normalization
        y3 = pp.normalize(y2, method=nm.method, peak_target=nm.peak_target, rms_target_db=nm.rms_target_db,
                          target_duration_s=None, sr=sr)
        out = pp.save_wav(td / "final.wav", y3, sr, bit_depth=P.bit_depth)
        y, _ = pp.load_wav(out)
    info = {"durasi_asli_s": len(y0) / sr, "durasi_setelah_trim_s": len(y1) / sr, "n_segmen_batuk": len(segs),
            "trim_mulai_s": i0 / sr, "amplitudo_maks_asli": float(np.max(np.abs(y0)))}
    return {"y_raw": y0, "y_trim": y1, "y_final": y, "sr": sr, "segmen": segs, "info": info}


# ---------------------------------------------------------------------------
# Model ML klasik
# ---------------------------------------------------------------------------
def tabular_features(y: np.ndarray, sr: int, cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = fx.extract_tabular(y, sr, cfg)
    df = pd.DataFrame([feats])
    df, _ = fx.clean_tabular(df, list(feats))   # aturan NaN yang sama dengan Tahap 3
    return df


def load_ml(algo: str, eksperimen: str, cfg=None) -> dict:
    cfg = cfg or load_config()
    path = cfg.paths.models / f"ml_{algo}_{eksperimen}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ada — jalankan dulu scripts/04_train_ml.py")
    return joblib.load(path)


def predict_ml(model: dict, X: pd.DataFrame) -> tuple[float, float]:
    """Return (P(COVID-19), ambang)."""
    p = float(model["pipeline"].predict_proba(X[model["features"]])[:, 1][0])
    return p, float(model["threshold"])


# ---------------------------------------------------------------------------
# CNN (hanya Eksperimen A)
# ---------------------------------------------------------------------------
def load_cnn(feature: str = "logmel", eksperimen: str = "A", cfg=None) -> dict:
    """Muat checkpoint CNN seed utama (42). Return dict: model (eval, CPU), ckpt."""
    import torch

    from src import models_cnn as mc
    cfg = cfg or load_config()
    path = cfg.paths.models / f"cnn_{feature}_{eksperimen}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ada — jalankan dulu scripts/05_train_cnn.py")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = mc.build_model(ck["config"])
    model.load_state_dict(ck["model"])
    model.eval()
    return {"model": model, "ckpt": ck}


def cnn_input(y: np.ndarray, sr: int, cnn: dict, cfg=None) -> np.ndarray:
    """Spektrogram ternormalisasi (F, T') float32, sama persis dengan input CNN saat evaluasi."""
    cfg = cfg or load_config()
    ck = cnn["ckpt"]
    L, M = fx.extract_2d(y, sr, cfg)
    x = (L if ck["feature"] == "logmel" else M).astype(np.float16).astype(np.float32)  # disimpan float16 di Tahap 3
    tp = int(ck["time_pool"])
    if tp > 1:
        T = x.shape[1] // tp * tp
        x = x[:, :T].reshape(x.shape[0], -1, tp).mean(-1)
    x = x.astype(np.float16).astype(np.float32)
    mean, std = ck["normalizer"]["mean"], ck["normalizer"]["std"]
    return ((x - mean) / std).astype(np.float16).astype(np.float32)


def predict_cnn(x: np.ndarray, cnn: dict, with_cam: bool = False) -> tuple[float, float, np.ndarray | None]:
    """Return (P(COVID-19), ambang, peta Grad-CAM 0..1 atau None)."""
    import torch

    from src import models_cnn as mc
    xt = torch.from_numpy(np.ascontiguousarray(x))[None, None]
    thr = float(cnn["ckpt"]["threshold"])
    if with_cam:
        cam, p = mc.grad_cam(cnn["model"], xt)
        return p, thr, cam
    with torch.no_grad():
        p = float(torch.sigmoid(cnn["model"](xt)).item())
    return p, thr, None


def label(p: float, thr: float) -> str:
    return "COVID-19" if p >= thr else "healthy"
