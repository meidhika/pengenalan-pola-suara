# PROGRESS — Proyek PPKB (COUGHVID)

## Status tahap
| Tahap | Status | Tanggal |
|---|---|---|
| 1. Setup, ekstraksi, eksplorasi metadata | ✅ Selesai | 2026-11-26 |
| 2. Preprocessing audio & dataset eksperimen | ✅ Selesai | 2026-11-26 |
| 3. Ekstraksi fitur & visualisasi | ✅ Selesai | 2026-11-26 |
| 4. Klasifikasi | ⏳ Belum | |

---

## Tahap 1 — Setup proyek, ekstraksi dataset, eksplorasi metadata

### Lingkungan
- Python 3.11.2 (`.venv`), kernel Jupyter **`ppkb`** terdaftar.
- ffmpeg 9.1 (gyan.dev) tersedia di PATH.
- **Tidak ada GPU NVIDIA** (AMD Radeon terintegrasi) → PyTorch **2.15.0+cpu** dari index CPU.
  Training CNN di Tahap 4 harus dirancang ringan (CPU).
- Paket inti: numpy 1.26.4, pandas 2.3.3, librosa 0.10.2.post1, scikit-learn 1.5.2.
  Daftar lengkap: `reports/logs/pip_freeze.txt`.
- Ruang kosong drive D: ±68 GB sebelum ekstraksi.

### File yang dihasilkan
- `.gitignore`, `requirements.txt`, `config.yaml`, `src/config.py` (loader + `set_seed`)
- `scripts/01_extract.py` — inspeksi zip, ekstraksi idempoten (`--limit`, `--inspect-only`), tangani zip bersarang
- `src/data_utils.py` — `load_metadata()`, `uuid_to_audio_path()`, `probe_audio()`, `load_audio_ffmpeg()`, helper label ahli (`expert_long`, `expert_labeled_mask`)
- `src/plots.py` — `save_fig`, `save_table`, `bar_counts`
- `notebooks/01_eksplorasi_metadata.ipynb` (sudah dieksekusi, output tersimpan)
- `reports/figures/01_*.png` (5 gambar), `reports/tables/01_*.csv` (10 tabel)
- `reports/logs/extract_summary.txt`
- `references/references.bib` (R0–R4)

### Hasil ekstraksi
- Zip **tidak bersarang**; isi langsung di folder `coughvid_20211012/` → `data/raw/coughvid_20211012/`.
- 34.434 `.json` + 34.434 audio: **29.348 `.webm`**, **1.777 `.ogg`**, **3.309 `.wav`**; `metadata_compiled.csv` (52 kolom, 34.434 baris).
- Semua 34.434 uuid di metadata punya file audio (1 file audio per uuid).

### Angka kunci
| Item | Jumlah |
|---|---|
| Total rekaman | 34.434 |
| `status`: healthy / symptomatic / COVID-19 / kosong | 15.476 / 3.873 / 1.315 / 13.770 |
| `status_SSL`: healthy / COVID-19 / kosong | 8.046 / 285 / 26.103 |
| Lolos `cough_detected ≥ 0.8` | 18.544 (53,9%) |
| **Populasi Eksperimen A** (≥0.8): COVID-19 / healthy | **720 / 10.132** (rasio ≈ 1 : 14) |
| symptomatic yang lolos ≥0.8 (tidak dipakai di A) | 2.683 |
| Rekaman berlabel ahli (≥1 `diagnosis_*`) | 2.841 (3.224 anotasi) |
| Nilai unik `diagnosis_*` | COVID-19, healthy_cough, lower_infection, obstructive_disease, upper_infection |
| Distribusi anotasi diagnosis | upper_infection 883, healthy_cough 746, lower_infection 732, COVID-19 649, obstructive_disease 214 |
| Eksperimen B (per rekaman): ≥1 ahli COVID-19 / ≥1 ahli healthy_cough / konflik keduanya | 621 / 705 / 34 |
| Eksperimen B tanpa konflik & ≥0.8: COVID-19 / healthy_cough | 504 / 547 |

Catatan penting untuk laporan (keandalan label, mendukung R4): crosstab `status` × diagnosis ahli
menunjukkan kesesuaian rendah — dari 810 anotasi pada rekaman self-reported COVID-19, hanya 161
didiagnosis COVID-19 oleh ahli (212 dinilai healthy_cough). Lihat `reports/tables/01_crosstab_status_x_diagnosis_ahli.csv`.

### Temuan teknis (penting untuk Tahap 2)
1. **Ekstensi `.wav` menyesatkan**: dari sampel 30 file `.wav`, 16 berisi codec **AAC**, sisanya
   PCM 16-bit; sample rate bervariasi (16 / 44,1 / 48 kHz). `.webm` & `.ogg` berisi **Opus 48 kHz mono**.
   → **Selalu dekode via ffmpeg** (`du.load_audio_ffmpeg`), jangan `soundfile.read` langsung.
2. Header `.webm` sering tanpa durasi → durasi dihitung dari jumlah sampel hasil dekode.
3. Durasi sampel 1,7–10,0 s (batas aplikasi perekam ±10 s) — sesuai `max_duration_s = 10`.
4. Kolom demografi (`age`, `gender`, dst.) kosong pada 13.770 baris yang sama dengan `status` kosong.

### Rekomendasi strategi sampling Tahap 2
- **Konversi ke wav 16 kHz mono** untuk: seluruh populasi A (720 COVID-19 + 10.132 healthy) +
  subset ahli Eksperimen B (±1.051 rekaman, sebagian tumpang-tindih). Total ≈ 11–12 ribu file,
  ±2–3 GB wav — masih wajar di CPU (estimasi puluhan menit dengan ffmpeg).
- **Split dulu** (stratified per `uuid`, 70/15/15, seed 42) pada populasi A **dengan rasio asli**;
  val/test tetap berdistribusi alami (≈1:14) agar evaluasi jujur.
- Ketidakseimbangan ditangani **hanya pada train**: (a) *class weight* / SMOTE untuk ML klasik,
  (b) *undersampling* healthy pada train ke rasio 1:3 (≈500 COVID : ≈1.500 healthy) untuk CNN
  di CPU, + augmentasi pada kelas COVID-19.
- Metrik utama: ROC-AUC & PR-AUC, sensitivitas/spesifisitas — bukan akurasi (baseline mayoritas ≈ 93%).

### Keputusan
- Ekstraksi mempertahankan struktur folder zip (`data/raw/coughvid_20211012/`); `find_raw_dir()` mencari otomatis.
- Penulis lengkap R4 di `references.bib` disingkat "and others" — verifikasi dari PDF sebelum laporan final.

### TODO / masalah terbuka
- `README.md` belum dibuat (bisa di tahap akhir).
- Tidak ada ID subjek → split per rekaman; catat sebagai keterbatasan (potensi rekaman ganda dari orang yang sama).


---

## Tahap 2 — Preprocessing audio & penetapan dataset eksperimen

### File yang dihasilkan
- `config.yaml` — bagian baru `data.healthy_to_covid_ratio`, `data.expert_apply_cough_threshold`,
  `data.experiments`, `split`, `preprocessing` (semua parameter di bawah).
- `src/data_utils.py` — `build_label_table()` (A/B/C), `make_splits()`, `undersample_train()`.
- `src/preprocessing.py` — `convert_to_wav()`, `trim_silence()`, `segment_coughs()`, `reduce_noise()`,
  `normalize()`, `preprocess_file()` (pipeline lengkap), `load_wav()`/`save_wav()`.
- `src/plots.py` — `plot_preprocessing_steps()`, `plot_spectrogram_compare()`.
- `scripts/02_preprocess.py` — joblib (n_jobs=-2), tqdm, resume (via `preprocess_meta.csv`), `--limit`,
  `--retry-failed`, `--resplit` (jangan dipakai: split final).
- `data/processed/splits_A.csv`, `splits_B.csv` — **SPLIT FINAL** (populasi penuh + kolom `split`, `dipakai`).
- `data/processed/manifest_A.csv`, `manifest_B.csv` — kolom: uuid, path_wav (relatif root), label, label_name,
  split, durasi_asli_s, durasi_trim_s, n_segmen, durasi_batuk_s, sr, cough_detected, status_self_reported
  (+ diagnosis_vote, n_ahli, ada_konflik untuk B).
- `data/processed/segments.csv` (batas tiap segmen batuk, detik relatif terhadap awal wav hasil trim),
  `data/processed/preprocess_meta.csv` (status semua file).
- `data/interim/*.wav` (5.881 file, 1,5 GB: hasil konversi), `data/processed/wav/*.wav` (5.864 file, 778 MB).
- `notebooks/02_preprocessing.ipynb` (dieksekusi), `reports/figures/02_*.png` (6), `reports/tables/02_*.csv` (3).
- Log: `reports/logs/preprocess_run.log`, `preprocess_errors.csv`, `preprocess_summary.json`, `label_stats_{A,B}.json`.

### Parameter preprocessing (urutan: a → b → c → d → e)
| Langkah | Parameter |
|---|---|
| a. Konversi (ffmpeg) | WAV mono, 16 kHz (Nyquist 8 kHz), PCM 16-bit (65.536 level) |
| b. Trim (`librosa.effects.trim`) | top_db = 30, frame 512, hop 128 |
| c. Segmentasi (adaptasi `segment_cough` COUGHVID, RMS per frame) | frame 25 ms, hop 10 ms, T_h = 2,0×RMS global, T_l = 0,5×RMS global, toleransi 30 ms, padding 0,2 s, min batuk 0,2 s |
| d. Reduksi noise (`noisereduce`) | spectral gating stasioner, prop_decrease = 0,8 |
| e. Normalisasi | peak → 0,99 (rentang −1..1); **tanpa padding saat disimpan** |
| Buang | durasi setelah trim < 0,3 s, audio kosong/senyap/rusak |

Segmentasi hanya disimpan sebagai **metadata** (`n_segmen`, `segments.csv`); wav hasil = sinyal penuh setelah trim
(jeda antar-batuk tetap ada). Pad/potong ke 10 s dilakukan saat memuat input CNN:
`normalize(y, target_duration_s=cfg.audio.max_duration_s)`.

### Penetapan populasi
- **A**: `status` ∈ {COVID-19, healthy}, `cough_detected ≥ 0,8` → 720 / 10.132.
- **B** (majority vote `diagnosis_1..4`, seri dibuang, + `cough_detected ≥ 0,8`):
  2.841 rekaman berlabel ahli; 130 dinilai >1 ahli, **124 di antaranya konflik** (kesepakatan antar ahli sangat
  rendah); 88 konflik terselesaikan oleh vote, **36 seri dibuang**; 34 rekaman berisi konflik langsung
  COVID-19 vs healthy_cough. Vote = COVID-19/healthy_cough: 1.223 → setelah filter cough_detected (−210): **470 / 543**.
- **C** (`status_SSL`, ≥0,8): 285 / 8.046 — fungsi tersedia (`build_label_table(df, "C")`), belum diproses audionya
  (jalankan `--experiments A B C` bila diperlukan).

### Jumlah data akhir (manifest, setelah preprocessing)
**Eksperimen A** (train di-*undersample* healthy:COVID = 3:1; val/test distribusi alami)

| split | COVID-19 | healthy | total | (populasi penuh split: COVID / healthy) |
|---|---|---|---|---|
| train | 502 | 1.507 | 2.009 | 504 / 7.092 |
| val | 108 | 1.514 | 1.622 | 108 / 1.520 |
| test | 108 | 1.516 | 1.624 | 108 / 1.520 |
| **total** | **718** | **4.537** | **5.255** | 720 / 10.132 |

**Eksperimen B** (tanpa undersampling, sudah seimbang)

| split | COVID-19 | healthy | total |
|---|---|---|---|
| train | 329 | 378 | 707 |
| val | 71 | 81 | 152 |
| test | 70 | 82 | 152 |
| **total** | **470** | **541** | **1.011** |

- File diproses: 5.881 uuid unik (gabungan A ∪ B; 402 uuid ada di keduanya). **Gagal/dibuang: 17** (semua
  "terlalu pendek setelah trim < 0,3 s"; tidak ada file rusak). OK: 5.864.
- Waktu proses: run penuh **258 s** untuk 5.831 file (+23 s uji 50 file) dengan 11 worker
  (≈0,5 s/file per core; dominan reduksi noise & trim).

### Ringkasan durasi (median, detik) — `reports/tables/02_ringkasan_durasi.csv`
| | asli | setelah trim | total segmen batuk |
|---|---|---|---|
| A COVID-19 | 9,78 | 4,40 | 1,43 |
| A healthy | 9,72 | 3,98 | 1,30 |
| B COVID-19 | 9,78 | 4,92 | 1,75 |
| B healthy | 9,60 | 3,17 | 0,99 |

### Keputusan & dampak undersampling (`healthy_to_covid_ratio = 3`)
- Undersampling **hanya pada train** setelah split → val/test tetap ≈1:14 (alami) sehingga ROC-AUC, PR-AUC,
  sensitivitas & spesifisitas di test tetap mencerminkan kondisi skrining nyata.
- Dampak: (1) komputasi turun ±3,5× (train 2.009 vs 7.596); (2) prior kelas di train (25% COVID) ≠ test (6,6%)
  → probabilitas keluaran model **terlalu tinggi (miskalibrasi)**; ambang 0,5 akan memberi sensitivitas lebih
  tinggi tetapi spesifisitas & *precision* lebih rendah. ROC-AUC relatif tidak terpengaruh prior; **PR-AUC/precision
  di test akan rendah** karena prevalensi rendah — wajar, laporkan bersama baseline prevalensi (≈0,066).
  Ambang keputusan sebaiknya dipilih di **val** (mis. Youden / sensitivitas target), bukan 0,5.
  (3) 5.585 rekaman healthy train dibuang → variasi kelas healthy berkurang; untuk ML klasik bisa dibandingkan
  dengan `ratio: null` + *class weight* (cukup jalankan ulang script; resume hanya memproses file baru,
  split tidak berubah).
- Split per `uuid` (tidak ada ID subjek). Split A dan B dibuat **independen** → 402 uuid ada di keduanya dengan
  split yang bisa berbeda. **Jangan** melatih di A lalu menguji di test B (atau sebaliknya) tanpa membuang uuid
  yang tumpang-tindih → leakage.

### Masalah / catatan untuk Tahap 3
1. **Segmentasi tidak sempurna**: 611/5.255 rekaman A (11,6%) dan 115/1.011 B tanpa segmen terdeteksi
   (`n_segmen = 0`); pada rekaman padat batuk, ambang 2×RMS global bisa melewatkan batuk yang lebih pelan
   (terlihat di `02_langkah_preprocessing_covid.png`). Karena itu fitur dihitung dari **wav penuh hasil trim**;
   `segments.csv` hanya opsional (fitur per segmen).
2. Sebagian rekaman asli > 10 s (maks 15,2 s) → saat input CNN dipotong ke 10 s.
3. **Potensi shortcut durasi di B**: healthy_cough lebih pendek setelah trim (median 3,2 vs 4,9 s). Fitur
   statistik harus berupa rata-rata/agregat ternormalisasi, bukan jumlah absolut; pertimbangkan melaporkan AUC
   fitur durasi saja sebagai baseline.
4. Reduksi noise stasioner hanya sedikit mengubah spektrogram (noise latar umumnya sudah rendah setelah trim).
5. Normalisasi peak per-rekaman → informasi kenyaringan absolut hilang (disengaja; gain mikrofon tidak seragam).


---

## Tahap 3 — Ekstraksi fitur & visualisasi representasi suara

### File yang dihasilkan
- `config.yaml` — bagian baru `features` (n_fft 512 = 32 ms, hop 160 = 10 ms, Hann; n_mfcc 13, n_gfcc 13 /
  40 filter gammatone, rolloff 0,85, contrast 5 pita dari 200 Hz, pyin 65–1000 Hz frame 1024; n_mels 64,
  n_mfcc_2d 40, top_db 80, dtype float16).
- `src/features.py` — `extract_tabular()`, `extract_2d()`, `stft_mag()`, `logmel()`, `gfcc_matrix()` (spafe 0.3.3),
  `pitch_pyin()`, `extract_file()` (cache .npz), `clean_tabular()`, `mann_whitney_table()` + `fdr_bh()`,
  `class_mean_2d()`, `feature_group()`.
- `src/plots.py` — `WARNA_KELAS` (COVID-19 oranye `#d1661f`, healthy biru `#2a6fb0`, aman buta warna),
  `plot_waveform_pair`, `plot_fft_frame`, `plot_tf_grid`, `plot_class_mean_logmel`, `plot_feature_violins`, `plot_projection`.
- `scripts/03_extract_features.py` — ProcessPool (11 worker), tqdm, resume via cache `data/features/cache/<uuid>.npz`,
  `--limit` (keluaran uji ke `data/features/uji_limit/`, sudah dihapus), `--overwrite`, `--experiments`.
- `data/features/`: `tabular_{A,B}.parquet`; `logmel_{A,B}_{train,val,test}.npy`, `mfcc2d_*`, `labels_*` (int8),
  `uuids_*` (urutan baris = urutan array). Total ±1,3 GB + cache ±1,3 GB (boleh dihapus; hanya untuk resume).
- `notebooks/03_fitur_dan_visualisasi.ipynb` (dieksekusi). `reports/figures/03_*.png` (7):
  waveform, spektrum FFT frame, representasi waktu–frekuensi (STFT/log-mel/MFCC/GFCC), rata-rata log-mel per kelas
  + selisih, violin fitur kunci, distribusi AUC fitur, proyeksi PCA & t-SNE.
- `reports/tables/03_uji_fitur.csv` (15 teratas A), `03_uji_fitur_semua_A.csv` (315 fitur), `03_uji_fitur_per_kelompok.csv`,
  `03_uji_fitur_B.csv` (15 teratas B).
- Log: `reports/logs/features_run.log`, `features_summary.json`, `features_nan_report.csv`, `nb03_exec.log`.
- `requirements.txt`: `pyarrow>=15,<18` (pyarrow 26 butuh NumPy 2 → diturunkan ke 17.0.0).

### Fitur tabular: 315 kolom fitur (+4 meta: uuid, split, label, label_name)
| Kelompok | Kolom | Isi |
|---|---|---|
| waktu | 9 | durasi_s; rms & zcr × (mean, std, min, max) |
| spektral | 40 | centroid, bandwidth, rolloff, flatness × 4 stat; contrast 6 pita × 4 stat |
| pitch | 6 | f0 × 4 stat (frame voiced saja), voiced_ratio, voiced_prob_mean |
| MFCC | 78 | 13 koef. × (mean, std, min, max, skew, kurt) |
| MFCC delta / delta-delta | 52 + 52 | 13 koef. × 4 stat |
| GFCC | 78 | 13 koef. × 6 stat |
Nama kolom: `<fitur>_<kk>_<stat>`, mis. `mfcc_03_mean`, `mfcc_d_00_min`, `contrast_05_std`.

### Fitur 2D (input CNN; 10 s, pad nol di akhir/potong; float16)
| Eksperimen | train | val | test |
|---|---|---|---|
| A log-mel | (2009, 64, 1001) | (1622, 64, 1001) | (1624, 64, 1001) |
| A MFCC-2D | (2009, 40, 1001) | (1622, 40, 1001) | (1624, 40, 1001) |
| B log-mel / MFCC-2D | (707, 64/40, 1001) | (152, …) | (152, …) |
Log-mel: dB, ref = 1,0, dipotong 80 dB di bawah puncak (padding = puncak − 80 dB). **Belum distandardisasi** —
normalisasi (mean/std per pita) harus di-*fit* pada train saja di Tahap 4.

### Proses & NaN/inf
- 5.864 uuid (A ∪ B), **0 error**, 5.279 s (±88 menit; bottleneck `librosa.pyin` default resolution 0,1 semitone,
  CPU 6 core fisik 100%). Uji `--limit 50`: 68 s, 0 error.
- **inf: 0.** Array 2D: semua finite. **NaN hanya di f0_mean/std/min/max** untuk rekaman tanpa frame voiced:
  739/5.255 (A, 14%) dan 144/1.011 (B) → **diisi 0** (aturan deterministik per rekaman, tanpa statistik dataset →
  bebas leakage; informasi "tidak ada pitch" tetap tercatat di `voiced_ratio = 0`). Tidak ada NaN lain.

### Temuan EDA fitur (split TRAIN saja)
**Eksperimen A (self-reported, 502 COVID vs 1.507 healthy)** — overlap kelas sangat besar:
- Hanya **39/315 fitur** signifikan (Mann-Whitney, p_FDR < 0,05); **tidak ada** efek sedang (|r| ≥ 0,28),
  hanya 10 fitur dengan efek kecil (|r| ≥ 0,11).
- Fitur terkuat: `contrast_05_std` (AUC 0,584, r = +0,17), `contrast_05_max` (0,574), `rms_min` (0,439),
  `durasi_s` (0,561, r = +0,12), `contrast_03_min`, `mfcc_d_00_min`, `mfcc_08_min`. Kelompok spektral paling
  informatif (18/40 signifikan); pitch tidak ada yang signifikan.
- Fitur "klasik" materi (ZCR, RMS mean, centroid, rolloff, voiced_ratio, f0, MFCC1–2 mean): AUC 0,48–0,52 → praktis
  tidak membedakan.
- PCA: 2 PC = 26,6% varians (10 PC 48%, 50 PC 76%); PCA & t-SNE menunjukkan kedua kelas **bercampur total**;
  silhouette (label sebagai klaster) ≈ 0 (−0,006 ruang penuh; +0,003 t-SNE).
- Rata-rata log-mel: selisih COVID − healthy hanya 0,4–1,6 dB per pita (0–4 s), dan pola selisih di 1–5 s
  terutama efek **durasi/padding** (COVID sedikit lebih panjang), bukan perbedaan timbre.
- Ekspektasi realistis Tahap 4: AUC model gabungan kemungkinan ≈ 0,60–0,65 untuk A.

**Eksperimen B (label ahli, 329 vs 378)** — perbedaan jauh lebih jelas: **178/315 fitur** signifikan; |r| ≈ 0,27–0,33
(efek sedang) untuk fitur dinamika MFCC (delta/delta-delta koef. 8–12, max/min) dan **`durasi_s` paling kuat
(AUC 0,667, r = +0,33)**. Mendukung R4 (label self-reported kurang andal), tetapi **durasi = potensi shortcut** →
di Tahap 4 laporkan baseline "durasi saja" dan model tanpa fitur durasi.

### Keputusan
- Statistik Mann-Whitney, pemilihan "fitur paling berbeda", rata-rata per kelas, dan PCA/t-SNE dihitung hanya pada
  **train** (val/test tidak disentuh) → aman dipakai untuk seleksi fitur di Tahap 4.
- Ukuran efek: rank-biserial r = 2·AUC − 1 (= Cliff's δ); koreksi FDR Benjamini–Hochberg (diimplementasi sendiri,
  statsmodels tidak terpasang).
- Fitur dihitung dari wav penuh hasil trim (bukan per segmen batuk) sesuai keputusan Tahap 2.
- Visual MFCC di grid waktu–frekuensi tidak menampilkan c0 (skala ≈ −400 menutupi koef. lain); fitur tabular tetap memuat c0.

### Masalah / TODO untuk Tahap 4
1. `durasi_s`, dan secara tidak langsung `rms_min`/padding pada array 2D, membawa informasi durasi → uji ablasi
   (dengan vs tanpa durasi; untuk CNN pertimbangkan *random crop*/mask padding).
2. `f0_*` saturasi di fmax 1000 Hz pada sebagian rekaman (terlihat di violin) — pitch batuk tidak andal; bobot rendah wajar.
3. Scaling (StandardScaler) & normalisasi log-mel di-fit pada train saja.
4. Tabular A hanya memuat train yang sudah di-*undersample* (2.009). Untuk ML klasik dengan semua healthy train
   (`ratio: null`), perlu jalankan ulang Tahap 2 lalu script ini (resume hanya menghitung uuid baru).
