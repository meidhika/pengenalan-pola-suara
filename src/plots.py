"""Utilitas grafik & tabel untuk laporan (PNG dpi 200, CSV)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import load_config

DPI = 200


def setup_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": DPI, "savefig.bbox": "tight"})


def save_fig(fig, name: str):
    """Simpan gambar ke reports/figures/<name>.png."""
    out = load_config().paths.figures / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    return out


def save_table(df: pd.DataFrame, name: str, index: bool = True):
    """Simpan tabel ke reports/tables/<name>.csv."""
    out = load_config().paths.tables / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index, encoding="utf-8")
    return out


def bar_counts(s: pd.Series, title: str, ax=None, dropna: bool = False, order=None):
    """Bar chart value_counts dengan label jumlah & persentase di atas batang."""
    vc = s.astype("object").fillna("(kosong)").value_counts() if not dropna else s.value_counts()
    if order is not None:
        vc = vc.reindex([o for o in order if o in vc.index])
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, color="#4C72B0")
    tot = vc.sum()
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v}\n({v / tot:.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Jumlah rekaman")
    ax.set_xlabel("")
    ax.margins(y=0.2)
    ax.tick_params(axis="x", rotation=20)
    return ax


# ============================================================================
# Tahap 2 — visualisasi preprocessing
# ============================================================================
def _time_axis(y, sr, offset_s: float = 0.0):
    import numpy as np

    return offset_s + np.arange(len(y)) / sr


def plot_preprocessing_steps(res: dict, title: str = "", sr: int = 16000):
    """Waveform tiap langkah preprocessing untuk satu rekaman.

    `res` = output `preprocessing.preprocess_file(..., return_signals=True)`.
    Panel: (1) asli + area yang dipertahankan trim, (2) setelah trim + kotak segmen batuk,
    (3) setelah reduksi noise, (4) setelah normalisasi peak.
    """
    from matplotlib.patches import Rectangle

    s = res["signals"]
    t0 = res["trim_mulai_s"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=False)
    ax = axes[0]
    ax.plot(_time_axis(s["asli"], sr), s["asli"], lw=0.5, color="#555555")
    ax.axvspan(t0, t0 + res["durasi_trim_s"], color="#4C72B0", alpha=0.15, label="dipertahankan (trim)")
    ax.set_title(f"(1) Waveform asli — {res['durasi_asli_s']:.2f} s, 16 kHz mono 16-bit")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    y = s["trim"]
    ax.plot(_time_axis(y, sr), y, lw=0.5, color="#4C72B0")
    h = max(1e-6, float(abs(y).max()))
    for i, (a, b) in enumerate(res["segments"]):
        ax.add_patch(Rectangle((a, -h * 1.05), b - a, 2.1 * h, fill=False, ec="#C44E52", lw=1.5,
                               label="segmen batuk" if i == 0 else None))
    ax.set_ylim(-h * 1.2, h * 1.2)
    ax.set_title(f"(2) Setelah trim ({res['durasi_trim_s']:.2f} s) + segmentasi: {res['n_segmen']} segmen batuk")
    if res["segments"]:
        ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(_time_axis(s["denoise"], sr), s["denoise"], lw=0.5, color="#55A868")
    ax.set_ylim(axes[1].get_ylim())
    ax.set_title("(3) Setelah reduksi noise (spectral gating)")

    ax = axes[3]
    ax.plot(_time_axis(s["final"], sr), s["final"], lw=0.5, color="#8172B2")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("(4) Setelah normalisasi peak (rentang −1..1)")
    for ax in axes:
        ax.set_ylabel("Amplitudo")
    axes[-1].set_xlabel("Waktu (s)")
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


def plot_spectrogram_compare(examples: dict, sr: int = 16000, n_fft: int = 512, hop: int = 128):
    """Spektrogram (dB) sebelum vs sesudah reduksi noise untuk beberapa contoh.

    `examples` = {judul: res} dengan res dari preprocess_file(return_signals=True).
    """
    import librosa
    import librosa.display
    import numpy as np

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.6 * n), squeeze=False, layout="constrained")
    for r, (name, res) in enumerate(examples.items()):
        for c, (key, lab) in enumerate([("trim", "sebelum reduksi noise"), ("final", "sesudah reduksi noise + normalisasi")]):
            y = res["signals"][key]
            y = y / max(1e-9, float(np.abs(y).max()))  # samakan skala agar adil dibandingkan
            S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)), ref=np.max)
            img = librosa.display.specshow(S, sr=sr, hop_length=hop, x_axis="time", y_axis="hz",
                                           ax=axes[r, c], vmin=-80, vmax=0, cmap="magma")
            axes[r, c].set_title(f"{name} — {lab}", fontsize=10)
            axes[r, c].set_xlabel("Waktu (s)"); axes[r, c].set_ylabel("Frekuensi (Hz)")
    fig.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.8)
    return fig
