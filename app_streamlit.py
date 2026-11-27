"""Dashboard Streamlit — demo klasifikasi suara batuk COVID-19 (COUGHVID v3).

Jalankan dari root proyek:
    .venv\\Scripts\\streamlit run app_streamlit.py
lalu buka http://localhost:8501. Input: unggah file audio ATAU rekam langsung lewat mikrofon browser.
Seluruh logika ada di src/inference.py (preprocessing & fitur identik dengan pipeline training).

BUKAN ALAT DIAGNOSIS MEDIS.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import librosa  # noqa: E402
import librosa.display  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import inference as inf  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.config import load_config  # noqa: E402

WARNA = {"COVID-19": "#E07B39", "healthy": "#257DE0"}   # sama dengan warna kelas di figur laporan
ANGGOTA = ["Meidhika Nawa Sapta (6250211011)", "Ari Risnanto (6250211006)", "Roslilie Simbulan (6250211001)"]

st.set_page_config(page_title="Demo Suara Batuk COVID-19", page_icon="🎙️", layout="wide")


# ---------------------------------------------------------------------------
# Sumber daya (di-cache sekali per proses server)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat konfigurasi & model…")
def muat_sumber():
    cfg = load_config()
    ml = {(a, e): inf.load_ml(a, e, cfg) for e in ("A", "B") for a in inf.NAMA_ML}
    cnn = {f: inf.load_cnn(f, "A", cfg) for f in inf.NAMA_CNN}
    return cfg, ml, cnn


@st.cache_data(show_spinner=False, max_entries=20)
def analisis(data: bytes, nama: str) -> dict:
    """Preprocessing + fitur tabular + log-mel untuk satu audio (di-cache per isi file)."""
    cfg, _, _ = muat_sumber()
    suffix = Path(nama).suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="ppkb_up_") as td:
        f = Path(td) / f"input{suffix}"
        f.write_bytes(data)
        st_ = inf.preprocess_stages(f, cfg)
    st_["X"] = inf.tabular_features(st_["y_final"], st_["sr"], cfg)
    return st_


def skor_semua(hasil: dict, cfg, ml, cnn, cam: bool) -> tuple[pd.DataFrame, dict]:
    rows, cams = [], {}
    for (algo, e), m in ml.items():
        p, thr = inf.predict_ml(m, hasil["X"])
        rows.append({"Eksperimen": e, "Model": inf.NAMA_ML[algo], "kunci": algo, "P(COVID-19)": p,
                     "Ambang": thr, "Prediksi": inf.label(p, thr)})
    for f, c in cnn.items():
        x = inf.cnn_input(hasil["y_final"], hasil["sr"], c, cfg)
        p, thr, cm = inf.predict_cnn(x, c, with_cam=cam)
        cams[f] = (x, cm)
        rows.append({"Eksperimen": "A", "Model": inf.NAMA_CNN[f], "kunci": f"cnn_{f}", "P(COVID-19)": p,
                     "Ambang": thr, "Prediksi": inf.label(p, thr)})
    df = pd.DataFrame(rows)
    df["Skor / ambang"] = df["P(COVID-19)"] / df["Ambang"]
    return df, cams


# ---------------------------------------------------------------------------
# Grafik
# ---------------------------------------------------------------------------
def fig_gauge(p: float, thr: float):
    fig, ax = plt.subplots(figsize=(7, 1.3))
    ax.barh([0], [1], color="#E8ECF1", height=0.5)
    ax.barh([0], [p], color=WARNA[inf.label(p, thr)], height=0.5)
    ax.axvline(thr, color="#102A3C", lw=2, ls="--")
    ax.text(thr, 0.42, f" ambang {thr:.3f}", va="bottom", ha="left", fontsize=9, color="#102A3C")
    dalam = p > 0.12   # label nilai di dalam batang bila cukup lebar
    ax.text(p - 0.01 if dalam else p + 0.01, 0, f"{p:.3f}", va="center", ha="right" if dalam else "left",
            fontsize=10, fontweight="bold", color="white" if dalam else "#102A3C")
    ax.set_xlim(0, 1); ax.set_ylim(-0.4, 0.8); ax.set_yticks([])
    ax.set_xlabel("P(COVID-19) — skor model (tidak terkalibrasi)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def fig_waveform(h: dict):
    sr = h["sr"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 4.2), sharex=False)
    t0 = np.arange(len(h["y_raw"])) / sr
    axs[0].plot(t0, h["y_raw"], lw=0.5, color="#888")
    a = h["info"]["trim_mulai_s"]
    axs[0].axvspan(a, a + len(h["y_trim"]) / sr, color="#257DE0", alpha=0.12, label="bagian dipertahankan (trim)")
    axs[0].set_title("1) Sinyal asli setelah konversi (16 kHz, mono, 16-bit) — domain waktu", fontsize=10)
    axs[0].legend(loc="upper right", fontsize=8); axs[0].set_ylabel("amplitudo")
    t1 = np.arange(len(h["y_final"])) / sr
    axs[1].plot(t1, h["y_final"], lw=0.5, color="#102A3C")
    for i, (s, e) in enumerate(h["segmen"]):
        axs[1].axvspan(s, e, color="#E07B39", alpha=0.18, label="segmen batuk" if i == 0 else None)
    axs[1].set_title("2) Setelah trim + reduksi noise + normalisasi peak (input ekstraksi fitur)", fontsize=10)
    if h["segmen"]:
        axs[1].legend(loc="upper right", fontsize=8)
    axs[1].set_xlabel("waktu (s)"); axs[1].set_ylabel("amplitudo")
    fig.tight_layout()
    return fig


def fig_spektral(h: dict, cfg):
    y, sr, F = h["y_final"], h["sr"], cfg.features
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=F.n_fft, hop_length=F.hop_length)), ref=np.max)
    im = librosa.display.specshow(D, sr=sr, hop_length=F.hop_length, x_axis="time", y_axis="hz", ax=axs[0],
                                  cmap="magma")
    axs[0].set_title("Spektrogram STFT (dB)", fontsize=10); fig.colorbar(im, ax=axs[0], format="%+.0f")
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=F.n_fft, hop_length=F.hop_length,
                                                           n_mels=F.n_mels), ref=np.max)
    im = librosa.display.specshow(S, sr=sr, hop_length=F.hop_length, x_axis="time", y_axis="mel", ax=axs[1],
                                  cmap="magma")
    axs[1].set_title(f"Log-mel spectrogram ({F.n_mels} pita)", fontsize=10); fig.colorbar(im, ax=axs[1])
    # FFT rata-rata seluruh rekaman → spektrum magnitudo (domain frekuensi)
    mag = np.abs(librosa.stft(y, n_fft=F.n_fft, hop_length=F.hop_length)).mean(axis=1)
    fr = librosa.fft_frequencies(sr=sr, n_fft=F.n_fft)
    axs[2].plot(fr, 20 * np.log10(mag + 1e-10), color="#102A3C", lw=1)
    axs[2].set_title("Spektrum magnitudo rata-rata (FFT)", fontsize=10)
    axs[2].set_xlabel("frekuensi (Hz)"); axs[2].set_ylabel("dB"); axs[2].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_mfcc(h: dict, cfg):
    y, sr, F = h["y_final"], h["sr"], cfg.features
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=F.n_mfcc, n_fft=F.n_fft, hop_length=F.hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=F.n_fft, hop_length=F.hop_length)[0]
    rms = librosa.feature.rms(y=y, frame_length=F.n_fft, hop_length=F.hop_length)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=F.hop_length)
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.4))
    im = librosa.display.specshow(M, sr=sr, hop_length=F.hop_length, x_axis="time", ax=axs[0], cmap="coolwarm")
    axs[0].set_title(f"MFCC ({F.n_mfcc} koefisien)", fontsize=10); axs[0].set_ylabel("koefisien")
    fig.colorbar(im, ax=axs[0])
    axs[1].plot(t, rms / (rms.max() + 1e-10), label="energi RMS (ternormalisasi)", color="#E07B39")
    axs[1].plot(t, zcr / (zcr.max() + 1e-10), label="zero-crossing rate (ternormalisasi)", color="#257DE0",
                alpha=0.8)
    axs[1].set_title("Energi & ZCR per frame (32 ms, hop 10 ms)", fontsize=10)
    axs[1].set_xlabel("waktu (s)"); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_gradcam(x: np.ndarray, cam: np.ndarray, durasi_s: float, hop_s: float):
    n = min(x.shape[1], int(np.ceil(min(durasi_s, 10) / hop_s)) + 5)   # tampilkan bagian berisi suara saja
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.4))
    ext = [0, n * hop_s, 0, x.shape[0]]
    axs[0].imshow(x[:, :n], origin="lower", aspect="auto", cmap="magma", extent=ext)
    axs[0].set_title("Input CNN: log-mel terstandardisasi", fontsize=10)
    axs[1].imshow(x[:, :n], origin="lower", aspect="auto", cmap="gray", extent=ext)
    axs[1].imshow(cam[:, :n], origin="lower", aspect="auto", cmap="jet", alpha=0.45, extent=ext)
    axs[1].set_title("Grad-CAM: area yang paling mendorong skor COVID-19", fontsize=10)
    for a in axs:
        a.set_xlabel("waktu (s)"); a.set_ylabel("pita mel")
    fig.tight_layout()
    return fig


def tampil(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Penjelasan Eksperimen A vs B (angka dari reports/tables: 02_jumlah_per_split_*, 06_tabel_final_*,
# 06_kesepakatan_label_B — sama dengan laporan)
# ---------------------------------------------------------------------------
TABEL_AB = pd.DataFrame({
    "Aspek": ["Sumber label", "Arti pertanyaan model", "Jumlah rekaman", "Data uji",
              "ROC-AUC data uji", "Model tersedia"],
    "Eksperimen A (utama)": [
        "Kolom `status`: pengakuan pengirim rekaman sendiri (*self-reported*)",
        "Apakah batuk ini mirip batuk orang yang **mengaku** COVID-19?",
        "5.255 (718 COVID-19 · 4.537 healthy); hanya train di-undersample 1:3",
        "1.624 rekaman, 108 COVID-19 (±6,7%)",
        "±0,57–0,58 (ML), 0,54–0,57 (CNN); acak = 0,52",
        "5 ML + 2 CNN"],
    "Eksperimen B": [
        "Kolom `diagnosis_*`: penilaian pulmonolog yang mendengarkan rekaman (*majority vote*)",
        "Apakah batuk ini mirip batuk yang **dinilai dokter** sebagai COVID-19?",
        "1.011 (470 COVID-19 · 541 healthy), subset berlabel ahli",
        "152 rekaman, 70 COVID-19",
        "±0,78–0,83 (ML); acak = 0,49",
        "5 ML (CNN tidak dilatih: data terlalu sedikit)"],
})


def penjelasan_ab(ringkas: bool = False):
    """Tampilkan perbedaan Eksperimen A dan B."""
    st.markdown("Kedua eksperimen memakai **suara dan 315 fitur yang sama**. Bedanya hanya pada **label** "
                "(\"kunci jawaban\") yang dipakai saat training: siapa yang menentukan sebuah rekaman batuk "
                "termasuk COVID-19 atau healthy.")
    st.dataframe(TABEL_AB, hide_index=True, width="stretch")
    st.markdown("""
**Mengapa B tampak jauh lebih bagus?** Bukan karena modelnya lebih pintar mendeteksi COVID-19:
- **Label A banyak *noise*:** status diisi sendiri tanpa bukti tes, sehingga tidak selalu sesuai kenyataan
  (sejalan dengan Xiong dkk., 2022).
- **Label B dari dokter yang mendengar suara batuknya:** model B belajar **meniru penilaian dokter dari suara**,
  tugas yang memang lebih "terdengar" di audio. Tetapi label B juga **bukan hasil PCR**.
- **Kedua label hampir tidak cocok:** pada 572 rekaman yang punya keduanya, kesepakatan hanya **50%** dengan
  **Cohen's κ = 0,00**, artinya pengakuan pengirim dan penilaian dokter praktis tidak berhubungan.
""")
    if not ringkas:
        st.markdown("Karena itu, model A dan B dapat memberi prediksi berbeda untuk rekaman yang sama. Hal ini "
                    "wajar, karena keduanya belajar dari kunci jawaban yang berbeda.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
cfg, ML, CNN = muat_sumber()
pilihan_model = {f"{inf.NAMA_ML[a]} — Eksperimen {e}": (a, e) for e in ("A", "B") for a in inf.NAMA_ML}
pilihan_model.update({f"{v} — Eksperimen A": (f"cnn_{k}", "A") for k, v in inf.NAMA_CNN.items()})

with st.sidebar:
    st.header("⚙️ Pengaturan")
    model_utama = st.selectbox("Model utama", list(pilihan_model), index=0,
                               help="Default: SVM Eksperimen A (ML terbaik menurut AUC validasi).")
    tampil_cam = st.checkbox("Hitung Grad-CAM (CNN log-mel)", value=True)
    st.divider()
    st.markdown("**Eksperimen A** — label *self-reported* `status` (COVID-19 vs healthy).  \n"
                "**Eksperimen B** — label diagnosis pulmonolog (subset berlabel ahli).  \n"
                "CNN hanya dilatih untuk Eksperimen A.",
                help="A: apakah batuk mirip batuk orang yang *mengaku* COVID-19? (AUC uji ±0,58)  \n"
                     "B: apakah batuk mirip batuk yang *dinilai dokter* sebagai COVID-19? (AUC uji ±0,83, "
                     "tetapi bukan terhadap PCR). Penjelasan lengkap ada di tab ℹ️ Tentang pipeline.")
    st.divider()
    st.caption("Kelompok — Pengenalan Pola & Kecerdasan Buatan (Pertemuan 8: Pengenalan Pola Suara)")
    for a in ANGGOTA:
        st.caption(f"• {a}")

# ---------------------------------------------------------------------------
# Halaman utama
# ---------------------------------------------------------------------------
st.title("🎙️ Klasifikasi Suara Batuk untuk Skrining COVID-19")
st.caption("Dataset COUGHVID v3 (Orlandic, Teijeiro & Atienza, 2021) · Akuisisi → Preprocessing → "
           "Ekstraksi fitur → Klasifikasi → Evaluasi")
st.warning(inf.DISCLAIMER, icon="⚠️")

tab_uji, tab_perf, tab_info = st.tabs(["🔬 Uji suara batuk", "📊 Performa model", "ℹ️ Tentang pipeline"])

def reset_input():
    """Ganti key widget unggah/rekam → widget dibuat ulang kosong, semua hasil di bawahnya hilang."""
    st.session_state["id_input"] = st.session_state.get("id_input", 0) + 1


st.session_state.setdefault("id_input", 0)
nid = st.session_state["id_input"]

def halaman_uji():
    """Isi tab Uji. Memakai return (bukan st.stop) agar tab lain tetap dirender."""
    st.subheader("1. Masukkan suara batuk")
    sumber = st.radio("Sumber audio", ["📁 Unggah file", "🎤 Rekam langsung (voice note)"], horizontal=True,
                      label_visibility="collapsed", on_change=reset_input)
    data, nama = None, None
    if sumber.startswith("📁"):
        up = st.file_uploader("Unggah rekaman batuk", type=inf.FORMAT_AUDIO, key=f"up_{nid}",
                              help="Format: " + ", ".join(inf.FORMAT_AUDIO) + ". Disarankan 1–10 detik, "
                                   "berisi 1–3 kali batuk.")
        if up is not None:
            data, nama = up.getvalue(), up.name
    else:
        st.info("Klik ikon mikrofon, izinkan akses mikrofon di browser, batuk 1–3 kali (±3–10 detik), "
                "lalu klik stop. Rekam di ruangan yang cukup tenang, jarak mulut–mikrofon ±20–30 cm.", icon="🎤")
        rek = st.audio_input("Rekam suara batuk", sample_rate=16000, key=f"rek_{nid}")
        if rek is not None:
            data, nama = rek.getvalue(), "rekaman.wav"

    if data is None:
        return

    c_audio, c_hapus = st.columns([4, 1], vertical_alignment="center")
    c_audio.audio(data)
    c_hapus.button("🗑️ Hapus & mulai baru", on_click=reset_input, type="primary", width="stretch",
                   help="Kosongkan audio dan semua hasil analisis, lalu unggah/rekam suara batuk baru.")
    try:
        with st.spinner("Preprocessing & ekstraksi 315 fitur…"):
            H = analisis(data, nama)
    except pp.PreprocessError as e:
        st.error(f"Audio ditolak oleh tahap preprocessing: {e}. Coba rekam ulang dengan suara batuk yang "
                 "lebih jelas.")
        return
    except Exception as e:  # noqa: BLE001 — tampilkan ke pengguna, jangan crash
        st.error(f"Gagal memproses audio: {e}")
        return

    with st.spinner("Menjalankan 12 model…"):
        df, cams = skor_semua(H, cfg, ML, CNN, cam=tampil_cam)

    info = H["info"]
    if info["n_segmen_batuk"] == 0:
        st.warning("Tidak ada segmen batuk terdeteksi — pastikan rekaman berisi suara batuk, bukan bicara/hening.")
    if info["durasi_asli_s"] > cfg.audio.max_duration_s + 0.5:
        st.info(f"Rekaman {info['durasi_asli_s']:.1f} s > {cfg.audio.max_duration_s} s: CNN hanya memakai "
                f"{cfg.audio.max_duration_s} s pertama (data training umumnya ≤ 10 s).")

    # ----- 2. Hasil model utama
    st.subheader("2. Hasil prediksi")
    with st.expander("❓ Apa beda Eksperimen A dan B?"):
        penjelasan_ab(ringkas=False)
    kunci, eks = pilihan_model[model_utama]
    r = df[(df["kunci"] == kunci) & (df["Eksperimen"] == eks)].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", r["Model"], f"Eksperimen {eks}", delta_color="off")
    c2.metric("P(COVID-19)", f"{r['P(COVID-19)']:.3f}")
    c3.metric("Ambang (Youden, data validasi)", f"{r['Ambang']:.3f}")
    ikon = "🟠" if r["Prediksi"] == "COVID-19" else "🔵"
    c4.metric("Prediksi model", f"{ikon} {r['Prediksi']}")
    tampil(fig_gauge(r["P(COVID-19)"], r["Ambang"]))
    st.caption("Label = COVID-19 bila skor ≥ ambang. Skor tidak terkalibrasi (data train di-undersample 1:3), "
               "jadi bandingkan dengan ambang, bukan dengan 0,5.")

    n_cov = int((df["Prediksi"] == "COVID-19").sum())
    st.markdown(f"**Konsensus:** {n_cov} dari {len(df)} model memprediksi *COVID-19*, "
                f"{len(df) - n_cov} memprediksi *healthy*.")
    tabel = df.drop(columns="kunci").copy()

    def warnai(v):
        return f"color: {WARNA.get(v, 'inherit')}; font-weight: 600"

    st.dataframe(tabel.style.format({"P(COVID-19)": "{:.3f}", "Ambang": "{:.3f}", "Skor / ambang": "{:.2f}"})
                 .map(warnai, subset=["Prediksi"]), hide_index=True, width="stretch")

    # ----- 3. Preprocessing
    st.subheader("3. Preprocessing")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Durasi asli", f"{info['durasi_asli_s']:.2f} s")
    m2.metric("Setelah trim senyap", f"{info['durasi_setelah_trim_s']:.2f} s")
    m3.metric("Segmen batuk terdeteksi", info["n_segmen_batuk"])
    m4.metric("Sampling / kuantisasi", f"{H['sr'] // 1000} kHz · {cfg.preprocessing.bit_depth}-bit")
    tampil(fig_waveform(H))
    a1, a2 = st.columns(2)
    a1.caption("Audio asli (setelah konversi 16 kHz mono)"); a1.audio(H["y_raw"], sample_rate=H["sr"])
    a2.caption("Audio setelah preprocessing (yang dianalisis model)"); a2.audio(H["y_final"], sample_rate=H["sr"])

    # ----- 4. Fitur
    st.subheader("4. Representasi & ekstraksi fitur")
    t1, t2, t3, t4 = st.tabs(["Spektrogram & FFT", "MFCC, energi & ZCR", "Grad-CAM CNN", "Vektor fitur (315)"])
    with t1:
        tampil(fig_spektral(H, cfg))
        st.caption("Batuk didominasi komponen **nonperiodik** (energi menyebar di banyak frekuensi, ledakan "
                   "singkat), berbeda dengan vokal yang periodik dan memiliki harmonik jelas.")
    with t2:
        tampil(fig_mfcc(H, cfg))
    with t3:
        x, cm = cams["logmel"]
        if cm is None:
            st.info("Aktifkan **Hitung Grad-CAM** di sidebar.")
        else:
            hop_s = cfg.features.hop_length / H["sr"] * CNN["logmel"]["ckpt"]["time_pool"]
            tampil(fig_gradcam(x, cm, info["durasi_setelah_trim_s"], hop_s))
            st.caption("Grad-CAM (Selvaraju dkk., 2017) terhadap logit COVID-19 pada blok konvolusi terakhir "
                       "LightCNN. Warna merah = kontribusi terbesar terhadap skor.")
    with t4:
        X = H["X"].T.rename(columns={0: "nilai"})
        X["kelompok"] = [c.split("_")[0] for c in X.index]
        st.dataframe(X, width="stretch", height=360)
        st.download_button("⬇️ Unduh fitur (CSV)", H["X"].to_csv(index=False).encode("utf-8"),
                           file_name="fitur_batuk.csv", mime="text/csv")


with tab_uji:
    halaman_uji()

# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader("Performa pada data uji (distribusi alami, tanpa undersampling)")
    kol = {"model": "Model", "n": "n", "n_pos": "n COVID", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
           "balanced_accuracy": "Balanced acc.", "recall_sensitivity": "Sensitivitas",
           "specificity": "Spesifisitas", "precision": "Precision", "f1": "F1", "threshold": "Ambang"}
    for e, judul in (("A", "Eksperimen A — label self-reported"), ("B", "Eksperimen B — label pulmonolog")):
        p = cfg.paths.tables / f"06_tabel_final_{e}.csv"
        if p.exists():
            t = pd.read_csv(p)
            t = t[[c for c in kol if c in t.columns]].rename(columns=kol)
            st.markdown(f"**{judul}**")
            st.dataframe(t.style.format(precision=3), hide_index=True, width="stretch")
    st.caption("AUC 0,55–0,60 pada label *self-reported* sejalan dengan temuan Xiong dkk. (2022): label "
               "crowdsourced tidak andal. Precision rendah karena prevalensi COVID-19 di data uji hanya ±6,7%.")
    g1, g2 = st.columns(2)
    for col, nama_fig, cap in ((g1, "06_roc_gabungan_A.png", "Kurva ROC gabungan — Eksperimen A"),
                               (g2, "04_roc_pr_B.png", "ROC & PR — Eksperimen B")):
        f = cfg.paths.figures / nama_fig
        if f.exists():
            col.image(str(f), caption=cap, width="stretch")

# ---------------------------------------------------------------------------
with tab_info:
    f = cfg.paths.figures / "07_alur_penelitian.png"
    if f.exists():
        st.image(str(f), caption="Alur penelitian", width="stretch")
    P = cfg.preprocessing
    st.markdown(f"""
**Langkah yang dijalankan dashboard ini pada setiap audio (identik dengan pipeline training):**

1. **Akuisisi** — file unggahan atau rekaman mikrofon browser (sinyal analog → digital oleh perangkat).
2. **Preprocessing**
   - konversi ffmpeg → WAV mono **{cfg.audio.sample_rate} Hz**, PCM **{P.bit_depth}-bit** (2¹⁶ = 65.536 level kuantisasi)
   - trim senyap (*top_db* {P.trim.top_db} dB), segmentasi batuk berbasis energi RMS
   - reduksi noise (*spectral gating* stasioner, *prop_decrease* {P.noise_reduction.prop_decrease})
   - normalisasi *peak* ke {P.normalization.peak_target}
3. **Ekstraksi fitur** — 315 fitur statistik (energi, ZCR, MFCC + Δ + ΔΔ, GFCC, fitur spektral, kontras,
   pitch/f0) untuk ML klasik; log-mel / MFCC 2D (10 s) untuk CNN.
4. **Klasifikasi** — pipeline tersimpan (`models/*.joblib`: scaler → SelectKBest → klasifikator) dan LightCNN
   (`models/cnn_*_A.pt`).
5. **Keputusan** — skor dibandingkan dengan ambang Youden yang dipilih di data validasi.

Kode: logika di `src/inference.py`, antarmuka di `app_streamlit.py`. Versi baris perintah: `scripts/predict.py`.
""")

    st.divider()
    st.subheader("Perbedaan Eksperimen A dan B")
    penjelasan_ab(ringkas=False)
