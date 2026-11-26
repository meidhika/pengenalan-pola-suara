"""Preprocessing audio batuk COUGHVID (Tahap 2 pipeline: akuisisi → PREPROCESSING → fitur → klasifikasi).

Urutan langkah yang dipakai `preprocess_file()` (dan scripts/02_preprocess.py):

    file asli (.webm/.ogg/.wav)
      └─ a. convert_to_wav()   → WAV mono 16 kHz PCM 16-bit           (data/interim/<uuid>.wav)
          └─ b. trim_silence() → buang hening awal/akhir
              └─ c. segment_coughs() → batas tiap batuk (metadata, data/processed/segments_*.csv)
                  └─ d. reduce_noise()   → spectral gating (opsional, config)
                      └─ e. normalize()  → peak/RMS normalization   (data/processed/wav/<uuid>.wav)

Semua parameter diambil dari `config.yaml` bagian `audio` dan `preprocessing`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from src.config import load_config


class PreprocessError(Exception):
    """Kesalahan yang menyebabkan rekaman dibuang (file rusak, kosong, terlalu pendek)."""


# ---------------------------------------------------------------------------
# a. Konversi format + resampling + kuantisasi
# ---------------------------------------------------------------------------
def convert_to_wav(src: str | Path, dst: str | Path, sr: int = 16000, bit_depth: int = 16,
                   overwrite: bool = False) -> Path:
    """Dekode file audio (webm/ogg/wav) dengan ffmpeg → WAV mono, `sr` Hz, PCM `bit_depth`-bit.

    Kaitan dengan materi *sampling & quantization*:

    - Suara asli adalah sinyal **analog** (kontinu dalam waktu dan amplitudo). Mikrofon
      smartphone + ADC mengubahnya menjadi sinyal **digital** dengan dua langkah:
    - **Sampling** (diskretisasi waktu): amplitudo diukur `sr` kali per detik. Dengan
      sr = 16.000 Hz, jarak antar-sampel = 1/16.000 s = 62,5 µs. Menurut teorema Nyquist,
      frekuensi tertinggi yang dapat direpresentasikan = sr/2 = **8 kHz**. Energi batuk dan
      formant suara manusia sebagian besar berada di bawah 8 kHz, sehingga 16 kHz memadai dan
      menghemat komputasi 3× dibanding rekaman asli (Opus 48 kHz). ffmpeg menerapkan filter
      *anti-aliasing* saat *resampling* sehingga komponen > 8 kHz tidak terlipat (*aliasing*).
    - **Quantization** (diskretisasi amplitudo): setiap sampel dibulatkan ke salah satu dari
      2^bit_depth level. PCM 16-bit → 2^16 = **65.536 level** (−32.768 … 32.767), rentang
      dinamis teoretis ≈ 6,02 × 16 ≈ **96 dB** — jauh di atas kebutuhan rekaman smartphone.
    - Kanal stereo (bila ada) dirata-ratakan menjadi **mono** (`-ac 1`).

    Catatan: ekstensi .wav di COUGHVID kadang berisi codec AAC, jadi SEMUA format didekode
    lewat ffmpeg (bukan dibaca langsung).

    Parameter
    ---------
    src, dst : path file sumber dan WAV tujuan.
    sr : sampling rate target (Hz).
    bit_depth : 16 (pcm_s16le) atau 32 (pcm_s32le).
    overwrite : bila False dan `dst` sudah ada → lewati (mendukung *resume*).

    Return
    ------
    Path WAV tujuan. Melempar PreprocessError bila ffmpeg gagal / hasil kosong.
    """
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 44 and not overwrite:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    codec = {16: "pcm_s16le", 32: "pcm_s32le"}[bit_depth]
    tmp = dst.with_suffix(".tmp.wav")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
           "-vn", "-ac", "1", "-ar", str(sr), "-c:a", codec, str(tmp)]
    pr = subprocess.run(cmd, capture_output=True, text=True)
    if pr.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 44:
        tmp.unlink(missing_ok=True)
        msg = (pr.stderr or "output kosong").strip().splitlines()
        raise PreprocessError(f"ffmpeg gagal: {msg[-1] if msg else 'output kosong'}")
    tmp.replace(dst)  # tulis atomik: file setengah jadi tidak dianggap selesai saat resume
    return dst


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Baca WAV PCM → float32 dalam rentang [-1, 1) beserta sampling rate."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def save_wav(path: str | Path, y: np.ndarray, sr: int, bit_depth: int = 16) -> Path:
    """Simpan float [-1, 1] sebagai WAV PCM (default 16-bit)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), np.clip(y, -1.0, 1.0), sr, subtype=f"PCM_{bit_depth}")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# b. Trimming hening
# ---------------------------------------------------------------------------
def trim_silence(y: np.ndarray, top_db: float = 30, frame_length: int = 512,
                 hop_length: int = 128) -> tuple[np.ndarray, tuple[int, int]]:
    """Buang bagian hening di awal dan akhir rekaman (`librosa.effects.trim`).

    Energi (RMS) dihitung per *frame* di domain waktu; frame dengan energi lebih rendah dari
    `top_db` dB di bawah frame paling keras dianggap hening. Hanya ujung awal/akhir yang
    dipotong — jeda di antara batuk dipertahankan (ditangani oleh segmentasi).

    Return
    ------
    (y_trim, (start, end)) — indeks sampel bagian yang dipertahankan pada sinyal asli.
    """
    import librosa

    if y.size == 0:
        return y, (0, 0)
    y_t, idx = librosa.effects.trim(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    return y_t, (int(idx[0]), int(idx[1]))


# ---------------------------------------------------------------------------
# c. Segmentasi batuk
# ---------------------------------------------------------------------------
def segment_coughs(y: np.ndarray, sr: int, frame_ms: float = 25, hop_ms: float = 10,
                   th_high_mult: float = 2.0, th_low_mult: float = 0.5, tolerance_ms: float = 30,
                   padding_s: float = 0.2, min_cough_s: float = 0.2) -> list[tuple[float, float]]:
    """Segmentasi batuk berbasis energi dengan *hysteresis threshold* (dua ambang).

    Diadaptasi dari fungsi `segment_cough` pada repositori resmi COUGHVID (Orlandic, Teijeiro &
    Atienza, 2021). Perbedaan: versi asli bekerja per sampel pada x²; versi ini bekerja pada
    **RMS per frame** (lebih cepat & lebih stabil terhadap noise) dengan ambang relatif terhadap
    RMS global rekaman:

    1. Hitung envelope RMS per frame (`frame_ms`, `hop_ms`).
    2. Ambang tinggi  T_h = th_high_mult × RMS_global → segmen DIMULAI bila RMS frame > T_h.
       Ambang rendah T_l = th_low_mult  × RMS_global → segmen DIAKHIRI bila RMS frame < T_l
       berturut-turut lebih dari `tolerance_ms`. Dua ambang (hysteresis) mencegah satu batuk
       terpecah-pecah akibat fluktuasi energi di sekitar satu ambang.
    3. Segmen dengan durasi inti < `min_cough_s` dibuang (klik/impuls pendek).
    4. Tambah `padding_s` di kiri/kanan (menangkap onset & ekor batuk), lalu gabungkan segmen
       yang saling tumpang-tindih.

    Batuk bersifat **nonperiodik** (ledakan udara turbulen, mirip noise lebar-pita) sehingga
    pendekatan berbasis energi lebih cocok daripada deteksi pitch/periodisitas.

    Return
    ------
    List (start_s, end_s) dalam detik, relatif terhadap awal `y`.
    """
    import librosa

    if y.size == 0:
        return []
    frame = max(1, int(round(sr * frame_ms / 1000)))
    hop = max(1, int(round(sr * hop_ms / 1000)))
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0]
    g = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if g <= 0:
        return []
    th_h, th_l = th_high_mult * g, th_low_mult * g
    tol = max(1, int(round(tolerance_ms / hop_ms)))

    raw: list[tuple[int, int]] = []
    in_seg, start, below = False, 0, 0
    for i, e in enumerate(rms):
        if not in_seg:
            if e > th_h:
                # mundur ke frame terakhir yang masih di atas T_l (onset sebenarnya)
                j = i
                while j > 0 and rms[j - 1] > th_l:
                    j -= 1
                in_seg, start, below = True, j, 0
        else:
            if e < th_l:
                below += 1
                if below > tol:
                    raw.append((start, i - below + 1))
                    in_seg = False
            else:
                below = 0
    if in_seg:
        raw.append((start, len(rms)))

    dur = len(y) / sr
    segs: list[list[float]] = []
    for s, e in raw:
        t0, t1 = s * hop / sr, e * hop / sr
        if t1 - t0 < min_cough_s:
            continue
        t0, t1 = max(0.0, t0 - padding_s), min(dur, t1 + padding_s)
        if segs and t0 <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], t1)
        else:
            segs.append([t0, t1])
    return [(round(a, 3), round(b, 3)) for a, b in segs]


# ---------------------------------------------------------------------------
# d. Reduksi noise
# ---------------------------------------------------------------------------
def reduce_noise(y: np.ndarray, sr: int, stationary: bool = True, prop_decrease: float = 0.8) -> np.ndarray:
    """Reduksi noise latar dengan *spectral gating* (pustaka `noisereduce`).

    Sinyal diubah ke domain frekuensi dengan STFT (Fourier per frame). Untuk tiap pita
    frekuensi diestimasi ambang noise (stasioner: rata-rata + k·std sepanjang rekaman). Bin
    spektrogram di bawah ambang dilemahkan sebanyak `prop_decrease`, lalu sinyal dikembalikan
    ke domain waktu dengan inverse STFT. Cocok untuk noise latar konstan (dengung, kipas, AC)
    yang umum pada rekaman smartphone crowdsourced.

    Diterapkan per rekaman secara independen → tidak ada *data leakage* antar split.
    """
    import noisereduce as nr

    if y.size < 2048:  # terlalu pendek untuk STFT yang bermakna
        return y
    out = nr.reduce_noise(y=y, sr=sr, stationary=stationary, prop_decrease=prop_decrease, n_jobs=1)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# e. Normalisasi amplitudo + panjang tetap
# ---------------------------------------------------------------------------
def normalize(y: np.ndarray, method: str = "peak", peak_target: float = 0.99, rms_target_db: float = -20,
              target_duration_s: float | None = None, sr: int = 16000) -> np.ndarray:
    """Normalisasi amplitudo dan (opsional) pad/potong ke durasi tetap.

    - method="peak": y / max|y| × peak_target → amplitudo maksimum ≈ ±1 (rentang -1..1).
      Menghilangkan perbedaan *gain* mikrofon/jarak antar perekam; bentuk gelombang tetap.
    - method="rms": skala agar RMS = rms_target_db dBFS (kenyaringan rata-rata seragam), lalu
      di-clip ke [-1, 1].
    - target_duration_s: bila diisi (mis. `audio.max_duration_s` = 10 s untuk input CNN),
      sinyal dipotong bila lebih panjang atau di-*zero-pad* di akhir bila lebih pendek,
      sehingga semua input berukuran sama (sr × durasi sampel).
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size:
        if method == "peak":
            m = float(np.max(np.abs(y)))
            if m > 0:
                y = y / m * peak_target
        elif method == "rms":
            r = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
            if r > 0:
                y = np.clip(y * (10 ** (rms_target_db / 20) / r), -1.0, 1.0)
        else:
            raise ValueError(f"metode normalisasi tidak dikenal: {method}")
    if target_duration_s is not None:
        n = int(round(target_duration_s * sr))
        y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline lengkap untuk satu rekaman
# ---------------------------------------------------------------------------
def preprocess_file(uuid: str, src: str | Path, cfg=None, overwrite: bool = False,
                    return_signals: bool = False) -> dict:
    """Jalankan langkah a–e untuk satu rekaman dan simpan WAV hasil.

    Output:
      data/interim/<uuid>.wav            (hasil konversi, 16 kHz mono 16-bit)
      data/processed/wav/<uuid>.wav      (trim + noise reduction + normalisasi)

    Return dict metadata: uuid, path_wav, sr, durasi_asli_s, durasi_trim_s, n_segmen,
    durasi_batuk_s, segments (list), status="ok". Melempar PreprocessError bila file
    rusak/kosong/terlalu pendek. Bila return_signals=True, sinyal tiap langkah ikut dikembalikan
    (untuk visualisasi notebook).
    """
    cfg = cfg or load_config()
    P, A = cfg.preprocessing, cfg.audio
    sr = A.sample_rate
    interim = cfg.paths.interim / f"{uuid}.wav"
    out_path = cfg.paths.processed / "wav" / f"{uuid}.wav"

    # a. konversi
    convert_to_wav(src, interim, sr=sr, bit_depth=P.bit_depth, overwrite=overwrite)
    y0, sr0 = load_wav(interim)
    if sr0 != sr:
        raise PreprocessError(f"sampling rate tak terduga {sr0}")
    if y0.size == 0 or not np.isfinite(y0).all():
        raise PreprocessError("audio kosong / tidak valid")
    if float(np.max(np.abs(y0))) < 1e-4:
        raise PreprocessError("audio senyap (amplitudo maks < 1e-4)")

    # b. trim
    y1, (i0, i1) = trim_silence(y0, **P.trim)
    if len(y1) / sr < P.min_duration_s:
        raise PreprocessError(f"terlalu pendek setelah trim ({len(y1) / sr:.2f} s < {P.min_duration_s} s)")

    # c. segmentasi (dari sinyal hasil trim)
    segs = segment_coughs(y1, sr, **P.segmentation)

    # d. reduksi noise
    nrc = P.noise_reduction
    y2 = reduce_noise(y1, sr, stationary=nrc.stationary, prop_decrease=nrc.prop_decrease) if nrc.enabled else y1

    # e. normalisasi
    nm = P.normalization
    tgt = A.max_duration_s if P.get("pad_on_save", False) else None
    y3 = normalize(y2, method=nm.method, peak_target=nm.peak_target, rms_target_db=nm.rms_target_db,
                   target_duration_s=tgt, sr=sr)
    save_wav(out_path, y3, sr, bit_depth=P.bit_depth)

    res = {
        "uuid": uuid, "path_wav": str(out_path), "sr": sr,
        "durasi_asli_s": round(len(y0) / sr, 3), "durasi_trim_s": round(len(y1) / sr, 3),
        "trim_mulai_s": round(i0 / sr, 3), "n_segmen": len(segs),
        "durasi_batuk_s": round(sum(b - a for a, b in segs), 3),
        "segments": segs, "status": "ok", "error": "",
    }
    if return_signals:
        res["signals"] = {"asli": y0, "trim": y1, "denoise": y2, "final": y3}
    return res
