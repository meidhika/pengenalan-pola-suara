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


# ===========================================================================
# Tahap 3 — visualisasi representasi suara & fitur
# ===========================================================================
import numpy as np  # noqa: E402

# Warna kelas (tervalidasi aman buta warna: ΔE CVD ≈ 22); dipakai konsisten di semua grafik Tahap 3+
WARNA_KELAS = {"COVID-19": "#d1661f", "healthy": "#2a6fb0"}
URUTAN_KELAS = ["healthy", "COVID-19"]


def plot_waveform_pair(examples: dict, sr: int = 16000):
    """Waveform (domain waktu) berdampingan. examples: {nama_kelas: y}."""
    fig, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 3.2), sharey=True)
    for ax, (k, y) in zip(np.atleast_1d(axes), examples.items()):
        t = np.arange(len(y)) / sr
        ax.plot(t, y, lw=0.5, color=WARNA_KELAS.get(k, "#555555"))
        ax.set(title=f"Waveform — {k} ({len(y) / sr:.2f} s)", xlabel="Waktu (s)", xlim=(0, t[-1]))
    np.atleast_1d(axes)[0].set_ylabel("Amplitudo (ternormalisasi)")
    fig.tight_layout()
    return fig


def plot_fft_frame(examples: dict, sr: int = 16000, n_fft: int = 512, hop: int = 160):
    """Spektrum FFT (domain frekuensi) satu frame 32 ms pada puncak energi batuk, per kelas.

    Baris atas: potongan waveform ±0,25 s di sekitar frame (frame diarsir). Baris bawah: |FFT| dB
    frame tersebut setelah jendela Hann.
    """
    import librosa

    fig, axes = plt.subplots(2, len(examples), figsize=(6 * len(examples), 6))
    axes = np.asarray(axes).reshape(2, -1)
    for j, (k, y) in enumerate(examples.items()):
        c = WARNA_KELAS.get(k, "#555555")
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, center=False)[0]
        s0 = int(np.argmax(rms)) * hop
        frame = y[s0:s0 + n_fft]
        a, b = max(0, s0 - sr // 4), min(len(y), s0 + n_fft + sr // 4)
        ax = axes[0, j]
        ax.plot(np.arange(a, b) / sr, y[a:b], lw=0.6, color=c)
        ax.axvspan(s0 / sr, (s0 + n_fft) / sr, color="#888888", alpha=0.3, label="frame 32 ms")
        ax.set(title=f"{k}: segmen di sekitar puncak energi", xlabel="Waktu (s)", ylabel="Amplitudo")
        ax.legend(loc="upper right", fontsize=8)
        X = np.fft.rfft(frame * np.hanning(len(frame)), n=n_fft)
        f = np.fft.rfftfreq(n_fft, 1 / sr)
        mag_db = 20 * np.log10(np.abs(X) + 1e-10)
        ax = axes[1, j]
        ax.plot(f, mag_db, lw=1, color=c)
        ax.set(title=f"{k}: spektrum FFT frame (resolusi {sr / n_fft:.2f} Hz)", xlabel="Frekuensi (Hz)",
               ylabel="Magnitudo (dB)", xlim=(0, sr / 2))
    fig.tight_layout()
    return fig


def plot_tf_grid(reps: dict, sr: int = 16000, hop: int = 160):
    """Grid representasi waktu-frekuensi: baris = representasi, kolom = kelas.

    reps: {nama_kelas: {"Spectrogram STFT (dB)": M, "Log-mel (dB)": M, "MFCC": M, "GFCC": M}}
    Skala warna disamakan per baris agar kedua kelas dapat dibandingkan.
    """
    import librosa.display

    kelas = list(reps)
    rows = list(reps[kelas[0]])
    fig, axes = plt.subplots(len(rows), len(kelas), figsize=(6.2 * len(kelas), 3.0 * len(rows)), squeeze=False,
                             layout="constrained")
    for i, r in enumerate(rows):
        vmin = min(np.percentile(reps[k][r], 1) for k in kelas)
        vmax = max(np.percentile(reps[k][r], 99) for k in kelas)
        for j, k in enumerate(kelas):
            M, ax = reps[k][r], axes[i, j]
            kw = dict(sr=sr, hop_length=hop, x_axis="time", ax=ax, vmin=vmin, vmax=vmax, cmap="magma")
            if r.startswith("Spectrogram"):
                img = librosa.display.specshow(M, y_axis="linear", **kw)
                ax.set_ylabel("Frekuensi (Hz)")
            elif r.startswith("Log-mel"):
                img = librosa.display.specshow(M, y_axis="mel", fmax=sr / 2, **kw)
                ax.set_ylabel("Frekuensi mel (Hz)")
            else:
                img = librosa.display.specshow(M, **kw)
                ax.set_ylabel("Indeks koefisien")
                ax.set_yticks(np.arange(M.shape[0]) + 0.5, labels=[str(v) for v in range(1, M.shape[0] + 1)]
                              if r.startswith("MFCC") else [str(v) for v in range(M.shape[0])], fontsize=6)
            ax.set_title(f"{r} — {k}")
            ax.set_xlabel("Waktu (s)")
        fig.colorbar(img, ax=axes[i, :].tolist(), shrink=0.9, pad=0.01)
    return fig


def plot_class_mean_logmel(mean_covid: np.ndarray, mean_healthy: np.ndarray, sr: int = 16000,
                           hop: int = 160, fmax: float = 8000):
    """Heatmap rata-rata log-mel per kelas + selisih (COVID-19 − healthy) dengan skala divergen."""
    import librosa.display

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    vmin = min(mean_covid.min(), mean_healthy.min())
    vmax = max(mean_covid.max(), mean_healthy.max())
    kw = dict(sr=sr, hop_length=hop, x_axis="time", y_axis="mel", fmax=fmax)
    for ax, M, t in ((axes[0], mean_covid, "Rata-rata log-mel — COVID-19"),
                     (axes[1], mean_healthy, "Rata-rata log-mel — healthy")):
        img = librosa.display.specshow(M, ax=ax, vmin=vmin, vmax=vmax, cmap="magma", **kw)
        ax.set(title=t, xlabel="Waktu (s)", ylabel="Frekuensi mel (Hz)")
    fig.colorbar(img, ax=axes[:2].tolist(), format="%+.0f dB", shrink=0.9, pad=0.01)
    D = mean_covid - mean_healthy
    lim = float(np.percentile(np.abs(D), 99))
    img = librosa.display.specshow(D, ax=axes[2], vmin=-lim, vmax=lim, cmap="RdBu_r", **kw)
    axes[2].set(title="Selisih (COVID-19 − healthy)", xlabel="Waktu (s)", ylabel="Frekuensi mel (Hz)")
    fig.colorbar(img, ax=axes[2], format="%+.1f dB", pad=0.01)
    return fig


def plot_feature_violins(df: pd.DataFrame, feats: list[str], label_col: str = "label_name",
                         uji: pd.DataFrame | None = None, ncols: int = 4):
    """Violin + boxplot per kelas untuk fitur kunci; judul memuat p (FDR) & rank-biserial r bila ada."""
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows), squeeze=False)
    info = uji.set_index("fitur") if uji is not None else None
    for ax, f in zip(axes.flat, feats):
        sns.violinplot(data=df, x=label_col, y=f, order=URUTAN_KELAS, hue=label_col, hue_order=URUTAN_KELAS,
                       palette=WARNA_KELAS, inner="box", cut=0, linewidth=0.8, ax=ax, legend=False)
        t = f
        if info is not None and f in info.index:
            r = info.loc[f]
            t += f"\nr = {r.rank_biserial_r:+.2f}, p_FDR = {r.p_fdr:.1e}"
        ax.set(title=t, xlabel="", ylabel="nilai fitur")
    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_projection(Zs: dict, labels: np.ndarray, names=("healthy", "COVID-19")):
    """Scatter proyeksi 2D (mis. {"PCA": Z, "t-SNE": Z}) diwarnai per kelas; COVID-19 digambar di atas."""
    fig, axes = plt.subplots(1, len(Zs), figsize=(6.5 * len(Zs), 5.5), squeeze=False)
    for ax, (t, Z) in zip(axes[0], Zs.items()):
        for lab, nm in enumerate(names):
            m = labels == lab
            ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=0.55 if lab == 0 else 0.8, color=WARNA_KELAS[nm],
                       edgecolors="white", linewidths=0.3, label=f"{nm} (n={m.sum()})")
        ax.set(title=t, xlabel="Komponen 1", ylabel="Komponen 2")
        ax.legend(loc="best", fontsize=9, markerscale=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tahap 4 — evaluasi model
# ---------------------------------------------------------------------------
# Warna model: urutan kategori tetap (tidak diputar ulang) + gaya garis sebagai penanda kedua
WARNA_MODEL = {
    "Logistic Regression": "#257de0", "SVM (RBF)": "#f47230", "k-NN": "#00b894",
    "Random Forest": "#f2a600", "XGBoost": "#ee7fac", "CNN": "#008a17",
    "Dummy (stratified)": "#8c8c8c", "CNN (log-mel)": "#008a17", "CNN (MFCC 2D)": "#7b4fd0",
}
GARIS_MODEL = {"Logistic Regression": "-", "SVM (RBF)": "--", "k-NN": "-.", "Random Forest": "-",
               "XGBoost": "--", "CNN": "-", "Dummy (stratified)": ":",
               "CNN (log-mel)": "-", "CNN (MFCC 2D)": "-."}


def plot_roc_pr_all(y_true, scores: dict, title: str = "", prevalence: float | None = None):
    """Kurva ROC dan PR semua model dalam satu figur (dua panel). scores: {nama: skor}."""
    from src.evaluate import curves
    from sklearn.metrics import average_precision_score, roc_auc_score

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for name, s in scores.items():
        c = curves(y_true, s)
        kw = dict(color=WARNA_MODEL.get(name, "#333"), ls=GARIS_MODEL.get(name, "-"), lw=2)
        axes[0].plot(c["fpr"], c["tpr"], label=f"{name} (AUC {roc_auc_score(y_true, s):.3f})", **kw)
        axes[1].plot(c["recall"], c["precision"], label=f"{name} (AP {average_precision_score(y_true, s):.3f})",
                     **kw)
    axes[0].plot([0, 1], [0, 1], color="#bbb", lw=1)
    axes[0].set(xlabel="False positive rate (1 − spesifisitas)", ylabel="True positive rate (sensitivitas)",
                title="Kurva ROC", xlim=(0, 1), ylim=(0, 1.01))
    prev = float(np.mean(y_true)) if prevalence is None else prevalence
    axes[1].axhline(prev, color="#bbb", lw=1)
    axes[1].text(0.99, prev, f"prevalensi = {prev:.3f}", ha="right", va="bottom", fontsize=8, color="#666")
    axes[1].set(xlabel="Recall (sensitivitas)", ylabel="Precision", title="Kurva Precision–Recall",
                xlim=(0, 1), ylim=(0, 1.01))
    for ax in axes:
        ax.legend(fontsize=8, loc="lower right" if ax is axes[0] else "upper right", frameon=True)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion(cm: np.ndarray, title: str = "", labels=("healthy", "COVID-19"), ax=None):
    """Confusion matrix dengan jumlah & persentase per baris (kelas sebenarnya)."""
    ax = ax or plt.subplots(figsize=(4.6, 4))[1]
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1%})", ha="center", va="center",
                    color="white" if pct[i, j] > 0.6 else "#222", fontsize=11)
    ax.set_xticks([0, 1], [f"pred. {l}" for l in labels])
    ax.set_yticks([0, 1], [f"aktual {l}" for l in labels])
    ax.grid(False)
    ax.set_title(title, fontsize=10)
    return ax


def plot_importance_top(imp: pd.DataFrame, title: str = "", top: int = 20, ax=None, color="#257de0"):
    """Bar horizontal top-N importance (kolom: fitur, importance, [std])."""
    d = imp.nlargest(top, "importance").iloc[::-1]
    ax = ax or plt.subplots(figsize=(6, 7))[1]
    xerr = d["std"] if "std" in d and d["std"].notna().any() else None
    ax.barh(d["fitur"], d["importance"], xerr=xerr, color=color, height=0.7,
            error_kw={"ecolor": "#666", "lw": 1})
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    return ax


def plot_ablation(abl: pd.DataFrame, title: str = ""):
    """Bar chart AUC test (dengan CI 95%) per subset fitur + AUC val & CV sebagai titik."""
    d = abl.reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.5))
    err = np.vstack([d["test_auc"] - d["test_auc_ci_low"], d["test_auc_ci_high"] - d["test_auc"]])
    ax.barh(y, d["test_auc"] - 0.5, left=0.5, xerr=err, color="#257de0", height=0.6,
            error_kw={"ecolor": "#444", "lw": 1, "capsize": 3}, label="AUC test (CI 95%)")
    ax.scatter(d["val_auc"], y, marker="D", s=40, color="#f47230", zorder=3, label="AUC val",
               edgecolor="white", linewidth=1)
    ax.scatter(d["cv_auc_mean"], y, marker="o", s=40, color="#00b894", zorder=3, label="AUC CV train",
               edgecolor="white", linewidth=1)
    ax.axvline(0.5, color="#999", lw=1)
    ax.set_yticks(y, [f"{s} (n={n})" for s, n in zip(d["subset"], d["n_fitur"])])
    ax.set_xlabel("ROC-AUC")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    return fig


# ---------------------------------------------------------------------------
# Tahap 5 — CNN
# ---------------------------------------------------------------------------
WARNA_SEED = {42: "#257de0", 7: "#f47230", 2024: "#00b894"}


def plot_training_curves(hist: pd.DataFrame, title: str = ""):
    """Kurva loss & AUC per epoch (train vs val) untuk tiap seed; titik = epoch terbaik (AUC val maks).

    Garis utuh = val, putus-putus = train (AUC train dihitung on-the-fly pada potongan teraugmentasi).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for seed, h in hist.groupby("seed"):
        c = WARNA_SEED.get(seed, "#555")
        axes[0].plot(h.epoch, h.train_loss, ls="--", color=c, lw=1.3, label=f"train (seed {seed})")
        axes[0].plot(h.epoch, h.val_loss, ls="-", color=c, lw=1.8, label=f"val (seed {seed})")
        axes[1].plot(h.epoch, h.train_auc, ls="--", color=c, lw=1.3, label=f"train (seed {seed})")
        axes[1].plot(h.epoch, h.val_auc, ls="-", color=c, lw=1.8, label=f"val (seed {seed})")
        b = h.loc[h.val_auc.idxmax()]
        axes[1].scatter(b.epoch, b.val_auc, color=c, s=45, zorder=3, edgecolor="white", linewidth=1)
    axes[0].set(xlabel="Epoch", ylabel="BCE loss (pos_weight)", title="Loss")
    axes[1].axhline(0.5, color="#bbb", lw=1)
    axes[1].set(xlabel="Epoch", ylabel="ROC-AUC", title="ROC-AUC (titik = checkpoint terbaik)")
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_gradcam_grid(items: list[dict], hop_s: float = 0.02, n_mels: int = 64, cols: int = 2):
    """Grid Grad-CAM. items: dict(spec (F,T), cam (F,T), title). Spektrogram abu-abu + overlay CAM."""
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.0 * rows), squeeze=False,
                             constrained_layout=True)
    for ax, it in zip(axes.flat, items):
        F, T = it["spec"].shape
        ext = [0, T * hop_s, 0, F]
        ax.imshow(it["spec"], origin="lower", aspect="auto", cmap="gray_r", extent=ext)
        im = ax.imshow(it["cam"], origin="lower", aspect="auto", cmap="inferno", alpha=0.45, extent=ext,
                       vmin=0, vmax=1)
        if it.get("length_s"):
            ax.axvline(it["length_s"], color="#00b894", lw=1.2, ls="--")
        ax.set_title(it["title"], fontsize=9)
        ax.set_xlabel("Waktu (s)")
        ax.set_ylabel("Indeks pita mel")
        ax.grid(False)
    for ax in list(axes.flat)[len(items):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Grad-CAM (0–1)")
    return fig


def plot_arch_diagram(blocks: list[tuple[str, str]], title: str = ""):
    """Diagram blok arsitektur horizontal. blocks: [(nama, keterangan bentuk/parameter), ...]."""
    from matplotlib.patches import FancyBboxPatch

    n = len(blocks)
    fig, ax = plt.subplots(figsize=(1.75 * n, 2.4))
    warna = ["#dfe9f6", "#cfe0f4", "#bcd3ef", "#a8c6ea", "#94b9e5", "#80ace0", "#fde3cf", "#f9c9a6"]
    for i, (nama, ket) in enumerate(blocks):
        x = i * 1.75
        ax.add_patch(FancyBboxPatch((x, 0), 1.45, 1.6, boxstyle="round,pad=0.03",
                                    fc=warna[min(i, len(warna) - 1)], ec="#456", lw=1))
        ax.text(x + 0.725, 1.15, nama, ha="center", va="center", fontsize=8.5, weight="bold")
        ax.text(x + 0.725, 0.55, ket, ha="center", va="center", fontsize=7.2, linespacing=1.3)
        if i < n - 1:
            ax.annotate("", xy=(x + 1.72, 0.8), xytext=(x + 1.5, 0.8),
                        arrowprops=dict(arrowstyle="->", color="#456", lw=1.2))
    ax.set_xlim(-0.1, n * 1.75)
    ax.set_ylim(-0.15, 1.8)
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    return fig
