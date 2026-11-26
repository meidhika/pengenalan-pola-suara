# TAHAP 3 — Ekstraksi Fitur dan Visualisasi Representasi Suara

> Ketik `/clear` dulu.

---

```
Baca CLAUDE.md dan PROGRESS.md. Kita mengerjakan TAHAP 3 saja: ekstraksi fitur dan visualisasi
representasi suara. Belum training model.

Langkah:
1. Buat `src/features.py` dengan dua kelompok fitur:
   A. Fitur tabular (vektor per rekaman, untuk ML klasik) — untuk tiap fitur frame-level, ambil
      statistik mean, std, min, max (dan skewness/kurtosis untuk MFCC):
      - Domain waktu: RMS energy, Zero Crossing Rate (ZCR), durasi.
      - Domain frekuensi/spektral: spectral centroid, bandwidth, rolloff, flatness, contrast.
      - Pitch-related: f0 (librosa.pyin) beserta rasio voiced — kaitkan dengan konsep periodik
        vs nonperiodik di materi.
      - Cepstral: MFCC 13 koefisien + delta + delta-delta.
      - GFCC 13 koefisien (spafe; cek API versi terpasang) — dasar dari R2/R3.
      Parameter frame: n_fft=512 (32 ms), hop=160 (10 ms) @16 kHz, taruh di config.yaml.
   B. Fitur 2D (untuk CNN): log-mel spectrogram (n_mels=64) dan matriks MFCC (n_mfcc=40) dengan
      ukuran tetap (sesuai max_duration_s). Simpan sebagai .npy per split (float16 bila perlu
      hemat ruang) di `data/features/`.
2. Buat `scripts/03_extract_features.py` (multiprocessing, tqdm, resume, --limit). Output:
   - `data/features/tabular_A.parquet` (uuid, split, label, + kolom fitur ber-nama jelas,
     mis. mfcc_03_mean). Lakukan juga untuk B.
   - `data/features/logmel_A_{train,val,test}.npy`, `mfcc2d_A_*.npy` + file label .npy.
   Uji --limit 50 dulu, lalu run penuh. Cek NaN/inf dan catat penanganannya.
3. Notebook `notebooks/03_fitur_dan_visualisasi.ipynb`, berisi visual yang memetakan materi kuliah:
   - Waveform (domain waktu) contoh COVID-19 vs healthy berdampingan.
   - Spektrum FFT (domain frekuensi) satu frame/segmen batuk.
   - Spectrogram (STFT) dan log-mel spectrogram, MFCC, dan GFCC untuk kedua kelas.
   - Rata-rata log-mel per kelas + selisihnya (heatmap).
   - Boxplot/violin fitur kunci per kelas (ZCR, RMS, centroid, beberapa MFCC) + uji Mann-Whitney U
     dan effect size; tabel 15 fitur paling berbeda → `reports/tables/03_uji_fitur.csv`.
   - Proyeksi 2D (PCA dan t-SNE pada sampel ≤ 3000) diwarnai per kelas.
   Semua figur → `reports/figures/03_*.png`, judul & label sumbu Bahasa Indonesia. Eksekusi notebook.
4. Perbarui PROGRESS.md: daftar fitur (jumlah total kolom), shape array 2D, temuan EDA fitur
   (fitur mana yang paling membedakan, seberapa besar overlap antar kelas). Berhenti dan ringkas.
```
