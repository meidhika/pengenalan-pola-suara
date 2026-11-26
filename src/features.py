"""Tahap 3 — ekstraksi fitur suara batuk.

Dua kelompok fitur (parameter di `config.yaml` bagian `features`):

A. Fitur TABULAR (satu vektor per rekaman, untuk ML klasik). Fitur *frame-level* dihitung dengan
   frame n_fft = 512 sampel (32 ms) dan hop 160 sampel (10 ms) @16 kHz, lalu diringkas menjadi
   statistik mean, std, min, max (+ skewness & kurtosis untuk MFCC dan GFCC):
   - Domain waktu      : RMS energy (amplitudo/energi), Zero Crossing Rate (ZCR), durasi.
   - Domain frekuensi  : spectral centroid, bandwidth, rolloff, flatness, contrast (dari |STFT|).
   - Pitch             : f0 dari `librosa.pyin` + rasio frame *voiced*. Batuk didominasi bunyi
                         NONPERIODIK (ledakan udara turbulen, mirip noise) → banyak frame tanpa f0;
                         bagian periodik (getaran pita suara pada fase "voiced") memberi f0.
   - Cepstral          : MFCC 13 + delta + delta-delta (skala mel, meniru persepsi telinga).
   - GFCC 13           : filterbank gammatone/ERB (spafe) — dasar fitur R2/R3.
B. Fitur 2D (input CNN), sinyal di-pad/dipotong ke `audio.max_duration_s` (10 s → 1001 frame):
   - log-mel spectrogram (n_mels = 64) dalam dB,
   - matriks MFCC (n_mfcc = 40) dari log-mel tersebut.

Nilai NaN (mis. statistik f0 bila tidak ada frame voiced) TIDAK diisi di sini; penanganannya
dilakukan dan dicatat oleh `clean_tabular()` (dipanggil di scripts/03_extract_features.py).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy import stats as sstats

from src.config import load_config
from src.preprocessing import load_wav, normalize

STATS_BASIC = ("mean", "std", "min", "max")
STATS_FULL = STATS_BASIC + ("skew", "kurt")


# ---------------------------------------------------------------------------
# Utilitas statistik
# ---------------------------------------------------------------------------
def _summ(x: np.ndarray, which=STATS_BASIC) -> dict:
    """Ringkas deret frame-level 1D menjadi statistik; deret kosong → NaN."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    out = {}
    for s in which:
        if x.size == 0:
            out[s] = np.nan
        elif s == "mean":
            out[s] = x.mean()
        elif s == "std":
            out[s] = x.std()
        elif s == "min":
            out[s] = x.min()
        elif s == "max":
            out[s] = x.max()
        elif s == "skew":
            out[s] = sstats.skew(x) if x.std() > 0 else np.nan
        elif s == "kurt":
            out[s] = sstats.kurtosis(x) if x.std() > 0 else np.nan
    return out


def _add(feats: dict, prefix: str, x: np.ndarray, which=STATS_BASIC):
    for k, v in _summ(x, which).items():
        feats[f"{prefix}_{k}"] = v


def _add_matrix(feats: dict, prefix: str, M: np.ndarray, which=STATS_BASIC):
    """M berbentuk (n_koef, n_frame) → kolom <prefix>_<kk>_<stat>, kk mulai 00."""
    for i in range(M.shape[0]):
        _add(feats, f"{prefix}_{i:02d}", M[i], which)


# ---------------------------------------------------------------------------
# Representasi dasar (juga dipakai notebook untuk visualisasi)
# ---------------------------------------------------------------------------
def stft_mag(y: np.ndarray, cfg=None) -> np.ndarray:
    """|STFT| (n_fft/2+1, n_frame) dengan jendela Hann 32 ms, hop 10 ms."""
    F = (cfg or load_config()).features
    return np.abs(librosa.stft(y, n_fft=F.n_fft, hop_length=F.hop_length, window=F.window))


def logmel(y: np.ndarray, sr: int, cfg=None) -> np.ndarray:
    """Log-mel spectrogram (n_mels, n_frame) dalam dB (ref = 1,0; dipotong top_db di bawah puncak)."""
    F = (cfg or load_config()).features
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=F.n_fft, hop_length=F.hop_length,
                                       window=F.window, n_mels=F.n_mels, fmin=F.fmin_mel,
                                       fmax=F.fmax_mel, power=2.0)
    return librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=F.top_db)


def mfcc_from_logmel(L: np.ndarray, n_mfcc: int) -> np.ndarray:
    """MFCC = DCT-II dari log-mel (cepstrum pada skala mel)."""
    return librosa.feature.mfcc(S=L, n_mfcc=n_mfcc)


def gfcc_matrix(y: np.ndarray, sr: int, cfg=None) -> np.ndarray:
    """GFCC (n_gfcc, n_frame) via spafe 0.3.x — filterbank gammatone skala ERB + log + DCT."""
    from spafe.features.gfcc import gfcc
    from spafe.utils.preprocessing import SlidingWindow

    F = (cfg or load_config()).features
    win = SlidingWindow(F.n_fft / sr, F.hop_length / sr, "hanning")
    G = gfcc(y.astype(np.float64), fs=sr, num_ceps=F.n_gfcc, nfilts=F.gfcc_nfilts, nfft=F.n_fft,
             window=win)
    return np.asarray(G).T  # spafe: (n_frame, n_ceps) → transpos agar seragam dengan librosa


def pitch_pyin(y: np.ndarray, sr: int, cfg=None):
    """f0 (Hz, NaN bila unvoiced), voiced_flag, voiced_prob dari probabilistic YIN."""
    F = (cfg or load_config()).features
    return librosa.pyin(y, fmin=F.pyin.fmin, fmax=F.pyin.fmax, sr=sr,
                        frame_length=F.pyin.frame_length, hop_length=F.hop_length)


# ---------------------------------------------------------------------------
# A. Fitur tabular
# ---------------------------------------------------------------------------
def extract_tabular(y: np.ndarray, sr: int, cfg=None) -> dict:
    """Vektor fitur statistik (dict nama → nilai) untuk satu rekaman (wav hasil Tahap 2, tanpa padding)."""
    cfg = cfg or load_config()
    F = cfg.features
    n_fft, hop = F.n_fft, F.hop_length
    feats: dict[str, float] = {}

    # --- Domain waktu ---
    feats["durasi_s"] = len(y) / sr
    _add(feats, "rms", librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0])
    _add(feats, "zcr", librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0])

    # --- Domain frekuensi (dari |STFT|, Fourier waktu-pendek) ---
    S = stft_mag(y, cfg)
    _add(feats, "centroid", librosa.feature.spectral_centroid(S=S, sr=sr)[0])
    _add(feats, "bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sr)[0])
    _add(feats, "rolloff", librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=F.rolloff_percent)[0])
    _add(feats, "flatness", librosa.feature.spectral_flatness(S=S)[0])
    C = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=F.contrast_n_bands, fmin=F.contrast_fmin)
    _add_matrix(feats, "contrast", C)

    # --- Pitch (periodik vs nonperiodik) ---
    f0, vflag, vprob = pitch_pyin(y, sr, cfg)
    _add(feats, "f0", f0[vflag])  # hanya frame voiced; kosong → NaN
    feats["voiced_ratio"] = float(np.mean(vflag)) if vflag.size else 0.0
    feats["voiced_prob_mean"] = float(np.nanmean(vprob)) if vprob.size else 0.0

    # --- Cepstral: MFCC + delta + delta-delta ---
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=F.n_mfcc, n_fft=n_fft, hop_length=hop,
                             window=F.window, n_mels=F.n_mels, fmax=F.fmax_mel)
    width = min(9, M.shape[1] if M.shape[1] % 2 else M.shape[1] - 1)
    _add_matrix(feats, "mfcc", M, STATS_FULL)
    _add_matrix(feats, "mfcc_d", librosa.feature.delta(M, width=width, order=1))
    _add_matrix(feats, "mfcc_dd", librosa.feature.delta(M, width=width, order=2))

    # --- GFCC ---
    _add_matrix(feats, "gfcc", gfcc_matrix(y, sr, cfg), STATS_FULL)
    return feats


# ---------------------------------------------------------------------------
# B. Fitur 2D (ukuran tetap)
# ---------------------------------------------------------------------------
def fix_length(y: np.ndarray, sr: int, cfg=None) -> np.ndarray:
    """Pad nol di akhir / potong ke audio.max_duration_s (tanpa mengubah amplitudo)."""
    cfg = cfg or load_config()
    n = int(round(cfg.audio.max_duration_s * sr))
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def extract_2d(y: np.ndarray, sr: int, cfg=None) -> tuple[np.ndarray, np.ndarray]:
    """(log-mel [n_mels, T], MFCC [n_mfcc_2d, T]) dengan T = 1 + sr·durasi/hop (1001 untuk 10 s)."""
    cfg = cfg or load_config()
    L = logmel(fix_length(y, sr, cfg), sr, cfg)
    return L, mfcc_from_logmel(L, cfg.features.n_mfcc_2d)


def shape_2d(cfg=None) -> dict:
    cfg = cfg or load_config()
    T = 1 + int(round(cfg.audio.max_duration_s * cfg.audio.sample_rate)) // cfg.features.hop_length
    return {"logmel": (cfg.features.n_mels, T), "mfcc2d": (cfg.features.n_mfcc_2d, T)}


# ---------------------------------------------------------------------------
# Satu file: hitung + simpan cache (untuk resume)
# ---------------------------------------------------------------------------
def extract_file(uuid: str, wav_path: str | Path, cache_dir: str | Path, overwrite: bool = False) -> dict:
    """Ekstrak fitur A & B satu rekaman → <cache_dir>/<uuid>.npz. Mengembalikan status ringkas."""
    cfg = load_config()
    out = Path(cache_dir) / f"{uuid}.npz"
    if out.exists() and not overwrite:
        return {"uuid": uuid, "status": "ada"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = load_wav(wav_path)
        if sr != cfg.audio.sample_rate:
            raise ValueError(f"sample rate {sr} != {cfg.audio.sample_rate}")
        tab = extract_tabular(y, sr, cfg)
        L, M2 = extract_2d(y, sr, cfg)
    dt = np.dtype(cfg.features.dtype_2d)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, names=np.array(list(tab.keys())), values=np.array(list(tab.values()), dtype=np.float64),
             logmel=L.astype(dt), mfcc2d=M2.astype(dt))
    tmp.replace(out)
    return {"uuid": uuid, "status": "ok"}


def load_cache(uuid: str, cache_dir: str | Path) -> dict:
    with np.load(Path(cache_dir) / f"{uuid}.npz") as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Pembersihan NaN/inf (dicatat)
# ---------------------------------------------------------------------------
def clean_tabular(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hitung & tangani NaN/inf pada fitur tabular. Mengembalikan (df_bersih, laporan).

    Aturan (deterministik per rekaman, tanpa statistik dataset → bebas leakage):
      - inf/-inf → NaN dulu.
      - Kolom f0_* NaN (rekaman tanpa frame voiced, voiced_ratio = 0) → 0 ("tidak ada pitch");
        informasinya tetap terbawa oleh voiced_ratio.
      - skew/kurt NaN (deret konstan) → 0 (distribusi simetris/tanpa ekor).
      - NaN lain → 0, dan dicatat (seharusnya tidak ada).
    """
    X = df[feature_cols].astype(np.float64)
    n_inf = np.isinf(X).sum()
    X = X.replace([np.inf, -np.inf], np.nan)
    n_nan = X.isna().sum()
    rep = pd.DataFrame({"n_inf": n_inf, "n_nan": n_nan})
    rep = rep[(rep.n_inf > 0) | (rep.n_nan > 0)].copy()

    def _rule(c):
        if c.startswith("f0_"):
            return "isi 0 (tidak ada frame voiced)"
        if c.endswith(("_skew", "_kurt")):
            return "isi 0 (deret konstan)"
        return "isi 0 (lain-lain)"

    rep["penanganan"] = [_rule(c) for c in rep.index]
    out = df.copy()
    out[feature_cols] = X.fillna(0.0).astype(np.float32)
    return out, rep


# ---------------------------------------------------------------------------
# Analisis beda fitur antar kelas (EDA Tahap 3)
# ---------------------------------------------------------------------------
def fdr_bh(p: np.ndarray) -> np.ndarray:
    """Koreksi Benjamini–Hochberg: p_adj(i) = min_{j≥i} p(j)·m/j (terurut naik), dibatasi ≤ 1."""
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


def mann_whitney_table(df: pd.DataFrame, feature_cols: list[str], label_col: str = "label") -> pd.DataFrame:
    """Uji Mann-Whitney U dua sisi per fitur (kelas 1 vs 0) + ukuran efek.

    - AUC fitur tunggal = U₁ / (n₁·n₀)  (peluang nilai COVID-19 > healthy).
    - rank-biserial r = 2·AUC − 1  (= Cliff's delta; |r| ≈ 0,11 kecil, 0,28 sedang, 0,43 besar).
    - p dikoreksi Benjamini–Hochberg (FDR) karena ratusan uji.
    """
    from scipy.stats import mannwhitneyu

    g1 = df[df[label_col] == 1]
    g0 = df[df[label_col] == 0]
    rows = []
    for c in feature_cols:
        a, b = g1[c].to_numpy(), g0[c].to_numpy()
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        auc = U / (len(a) * len(b))
        rows.append({"fitur": c, "median_covid": np.median(a), "median_healthy": np.median(b),
                     "U": U, "p_value": p, "auc_fitur": auc, "rank_biserial_r": 2 * auc - 1})
    res = pd.DataFrame(rows)
    res["p_fdr"] = fdr_bh(res["p_value"].to_numpy())
    res["abs_r"] = res["rank_biserial_r"].abs()
    res["signifikan_fdr_0.05"] = res["p_fdr"] < 0.05
    return res.sort_values("abs_r", ascending=False).reset_index(drop=True)


def feature_group(col: str) -> str:
    """Kelompok fitur dari nama kolom (untuk ringkasan)."""
    for pre, g in (("mfcc_dd", "MFCC delta-delta"), ("mfcc_d", "MFCC delta"), ("mfcc", "MFCC"),
                   ("gfcc", "GFCC"), ("contrast", "spektral"), ("centroid", "spektral"),
                   ("bandwidth", "spektral"), ("rolloff", "spektral"), ("flatness", "spektral"),
                   ("f0", "pitch"), ("voiced", "pitch"), ("rms", "waktu"), ("zcr", "waktu"),
                   ("durasi", "waktu")):
        if col.startswith(pre):
            return g
    return "lain"


def class_mean_2d(arr: np.ndarray, labels: np.ndarray, chunk: int = 256) -> dict[int, np.ndarray]:
    """Rata-rata array 2D (N, F, T) per label, dihitung per potongan (cocok untuk memmap float16)."""
    out = {}
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        acc = np.zeros(arr.shape[1:], dtype=np.float64)
        for i in range(0, len(idx), chunk):
            acc += arr[idx[i:i + chunk]].astype(np.float32).sum(axis=0)
        out[int(lab)] = (acc / len(idx)).astype(np.float32)
    return out
