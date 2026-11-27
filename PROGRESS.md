# PROGRESS — Proyek PPKB (COUGHVID)

> **STATUS PROYEK: ✅ SELESAI (2026-11-27).** Semua tahap 1–8 selesai. Paket pengumpulan ada di `submission/`.
> Sisa pekerjaan hanya pengisian/cek manual oleh kelompok — lihat **Tahap 8 → Checklist pengumpulan**.

## Status tahap
| Tahap | Status | Tanggal |
|---|---|---|
| 1. Setup, ekstraksi, eksplorasi metadata | ✅ Selesai | 2026-11-26 |
| 2. Preprocessing audio & dataset eksperimen | ✅ Selesai | 2026-11-26 |
| 3. Ekstraksi fitur & visualisasi | ✅ Selesai | 2026-11-26 |
| 4. Klasifikasi ML klasik (fitur tabular) | ✅ Selesai | 2026-11-26 |
| 5. CNN ringan (fitur 2D, Eksperimen A) | ✅ Selesai | 2026-11-26 |
| 6. Evaluasi komprehensif, perbandingan, analisis | ✅ Selesai | 2026-11-26 |
| 7a. Draf laporan per bab (Markdown) | ✅ Selesai | 2026-11-26 |
| 7b. Laporan Word + PDF | ✅ Selesai | 2026-11-26 |
| 8. Slide, demo, README, QA, paket pengumpulan | ✅ Selesai | 2026-11-27 |

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


---

## Tahap 4 — Klasifikasi dengan ML klasik (fitur tabular, Eksperimen A & B)

### File yang dihasilkan
- `src/evaluate.py` — metrik bersama untuk SEMUA model (dipakai juga oleh CNN): `compute_metrics()` (accuracy,
  balanced acc, precision, recall/sensitivitas, spesifisitas, F1, ROC-AUC, PR-AUC, TP/FP/TN/FN, CI 95% bootstrap
  1000× untuk ROC-AUC & PR-AUC), `youden_threshold()`, `threshold_metrics()`, `bootstrap_ci()`, `curves()`,
  `confusion()`, `save_results(model_name, experiment, metrics)` (upsert per model+eksperimen+split), `load_results()`.
- `src/models_ml.py` — pipeline imblearn `StandardScaler → SelectKBest(f_classif) → [SMOTE] → clf`;
  `make_pipeline()`, `param_space()`, `imbalance_variants()`, `feature_subsets()` (ablasi), `builtin_importance()`.
- `src/plots.py` — `WARNA_MODEL`/`GARIS_MODEL`, `plot_roc_pr_all`, `plot_confusion`, `plot_importance_top`, `plot_ablation`.
- `scripts/04_train_ml.py` — tqdm, log ke file, resume (model .joblib yang ada dimuat; `--overwrite` untuk latih
  ulang), `--limit N` (keluaran ke `reports/logs/uji_limit/`, sudah dihapus), `--n-iter`, `--skip-ablation`.
- `models/ml_<algo>_{A,B}.joblib` (dict: pipeline, threshold, fitur, ringkasan tuning, varian imbalance, prediksi
  val/test); `models/preds_ml_{A,B}.parquet` (prediksi val/test semua model → kurva ROC/PR; bisa digabung dengan CNN).
- `reports/tables/hasil_semua_model.csv` (12 baris: 6 model × A/B, split test), `04_imbalance_{A,B}.csv`,
  `04_ablasi_fitur.csv` (A), `04_ablasi_fitur_B.csv`, `04_importance_{A,B}.csv`.
- `notebooks/04_ml_klasik.ipynb` (dieksekusi). `reports/figures/04_{roc_pr,confusion,importance,ablasi}_{A,B}.png` (8).
- Log: `reports/logs/train_ml_{A,B}.log`, `train_ml_{A,B}_summary.json`, `train_ml_console.txt`, `nb04_exec.log`.
- Waktu: A 842 s, B 335 s (paling lama: SVM `probability=True` + SMOTE, dan RF).

### Protokol (keputusan)
- **Tuning di TRAIN saja** (opsi "train lalu val"): RandomizedSearchCV n_iter = 20, scoring ROC-AUC;
  A: StratifiedKFold 5; B: RepeatedStratifiedKFold 5×3. Grid termasuk `select__k` ∈ {20, 50, 100, all}.
- Imbalance: `balanced` (class_weight / scale_pos_weight XGBoost; k-NN: `none`) vs **SMOTE di dalam pipeline**
  → dipilih per algoritma berdasarkan AUC CV train.
- **VAL**: ambang Youden's J + pemilihan model terbaik antar-algoritma (AUC val). Model tidak dilatih ulang dengan val.
- **TEST**: dievaluasi sekali per model final. Variabilitas: repeated stratified CV 5×5 di train (mean ± std).
- Ablasi: algoritma terbaik + varian imbalance-nya, tuning ulang per subset (n_iter = 10). Tambahan subset
  (vii) durasi saja dan (viii) semua tanpa durasi (cek *shortcut*). "MFCC" = statis + Δ + ΔΔ (182 kolom);
  "waktu" = RMS & ZCR (8 kolom, tanpa durasi).
- Importance: bawaan RF (MDI) & XGBoost (gain) pada fitur yang lolos SelectKBest + **permutation importance di VAL**
  (ROC-AUC, 10 ulangan) untuk model terbaik.

### Hasil TEST (ambang Youden dari val)
**Eksperimen A** (n = 1.624, prevalensi 0,067)

| Model | imbalance | CV 5×5 AUC | AUC val | **AUC test [CI 95%]** | PR-AUC | Sens | Spes | F1 |
|---|---|---|---|---|---|---|---|---|
| Dummy (stratified) | – | 0,491 ± 0,023 | 0,476 | 0,516 [0,472–0,563] | 0,069 | 1,000 | 0,000 | 0,125 |
| Logistic Regression | balanced | 0,577 ± 0,033 | 0,561 | 0,569 [0,514–0,623] | 0,087 | 0,481 | 0,600 | 0,136 |
| **SVM (RBF)** ★ | balanced | 0,580 ± 0,034 | **0,574** | 0,576 [0,523–0,629] | 0,088 | 0,556 | 0,553 | 0,142 |
| k-NN | none | 0,564 ± 0,029 | 0,542 | 0,580 [0,525–0,636] | 0,088 | 0,509 | 0,613 | 0,147 |
| Random Forest | balanced | 0,583 ± 0,033 | 0,527 | 0,570 [0,513–0,628] | 0,091 | 0,407 | 0,705 | 0,147 |
| XGBoost | balanced | 0,576 ± 0,030 | 0,546 | 0,572 [0,513–0,629] | 0,097 | 0,630 | 0,433 | 0,131 |

**Eksperimen B** (label ahli; n = 152, prevalensi 0,461)

| Model | imbalance | CV 5×5 AUC | AUC val | **AUC test [CI 95%]** | PR-AUC | Sens | Spes | F1 |
|---|---|---|---|---|---|---|---|---|
| Dummy (stratified) | – | 0,487 ± 0,054 | 0,428 | 0,487 [0,417–0,567] | 0,454 | 1,000 | 0,000 | 0,631 |
| Logistic Regression | balanced | 0,767 ± 0,035 | 0,712 | 0,812 [0,730–0,877] | 0,749 | 0,771 | 0,817 | 0,777 |
| **SVM (RBF)** ★ | SMOTE | 0,766 ± 0,035 | **0,718** | **0,826 [0,745–0,888]** | 0,786 | 0,743 | 0,805 | 0,754 |
| k-NN | none | 0,723 ± 0,033 | 0,648 | 0,776 [0,689–0,848] | 0,675 | 0,629 | 0,756 | 0,657 |
| Random Forest | SMOTE | 0,748 ± 0,040 | 0,685 | 0,817 [0,741–0,879] | 0,771 | 0,800 | 0,732 | 0,757 |
| XGBoost | balanced | 0,753 ± 0,045 | 0,713 | 0,807 [0,726–0,871] | 0,754 | 0,614 | 0,817 | 0,672 |

★ = model terbaik menurut AUC val (dipakai untuk ablasi & permutation importance). A: SVM RBF (C ≈ 2,8,
γ ≈ 2,6e-4, semua fitur, class_weight balanced, ambang 0,249). B: SVM RBF + SMOTE (parameter sama, ambang 0,557).

### Ablasi fitur (SVM, AUC test; CV & val di CSV/figur)
| Subset | n | A | B |
|---|---|---|---|
| (i) waktu (RMS, ZCR) | 8 | 0,485 | 0,680 |
| (ii) spektral | 40 | 0,542 | 0,667 |
| (iii) MFCC (+Δ, ΔΔ) | 182 | 0,568 | 0,813 |
| (iv) GFCC | 78 | 0,495 | 0,764 |
| (v) MFCC + GFCC | 260 | 0,493 | 0,819 |
| (vi) semua | 315 | 0,576 | 0,826 |
| (vii) durasi saja | 1 | 0,518 | 0,671 |
| (viii) semua tanpa durasi | 314 | 0,576 | 0,823 |

### Fitur terpenting
- **A**: `contrast_05_max/std/mean` (kontras spektral pita ±3,2–6,4 kHz) peringkat 1 di RF, XGBoost, dan permutasi;
  lalu MFCC koefisien tengah–tinggi (`mfcc_06_mean`, `mfcc_11_mean`, `mfcc_09_max`, `mfcc_d_00_min`).
  Top-20 permutasi: 11 MFCC, 4 spektral. Konsisten dengan uji Mann-Whitney Tahap 3.
- **B**: permutasi (SVM) didominasi statistik bentuk MFCC (`mfcc_06_skew`, `mfcc_08_mean`, `mfcc_06_kurt`, `mfcc_00_kurt`)
  dan ΔΔ (`mfcc_dd_00_std`); RF/XGBoost menonjolkan dinamika MFCC (Δ/ΔΔ koef. 9–11), pitch (`f0_max`, `voiced_ratio`),
  GFCC, dan `durasi_s`.

### Catatan interpretasi (untuk laporan)
1. **A hanya sedikit di atas acak** (AUC 0,57–0,58; batas bawah CI > 0,5, tetapi CI semua model tumpang-tindih) →
   tidak ada algoritma yang terbukti unggul; precision ≈ 0,08 vs prevalensi 0,067 → **tidak layak** untuk skrining.
2. **B jauh lebih baik** (AUC 0,78–0,83) dengan fitur & pipeline yang sama → kualitas label adalah faktor pembatas
   utama (mendukung R4). Caveat: populasi B berbeda & seimbang, test kecil (CI ±0,07), AUC test > val/CV (variasi
   sampel kecil), label ahli = penilaian akustik pulmonolog, bukan PCR.
3. balanced vs SMOTE: selisih ≤ 0,01 AUC CV → tidak berpengaruh berarti pada AUC.
4. **Miskalibrasi akibat undersampling train** terlihat jelas: SVM A pada ambang 0,5 memprediksi **semua** rekaman
   sebagai healthy (sens 0); ambang Youden dari val (0,249) diperlukan (`04_confusion_A.png`, panel kanan).
5. **Durasi bukan shortcut bagi model penuh** (tanpa durasi: A 0,576 → 0,576; B 0,826 → 0,823), tetapi di B durasi
   saja sudah AUC 0,67 → sebut sebagai *confounder* potensial.
6. MFCC ≈ GFCC (redundan; MFCC sedikit lebih baik); fitur domain waktu (RMS/ZCR) paling lemah.
7. Dummy memberi sens 1 / spes 0 karena proba stratified bernilai 0/1 dan Youden di val memilih ambang 0 — wajar untuk
   baseline acak; yang relevan AUC ≈ 0,5 dan PR-AUC ≈ prevalensi.

### TODO / catatan untuk tahap CNN
- Pakai `src/evaluate.py` (`compute_metrics`, `youden_threshold` di val, `save_results`) agar baris CNN masuk
  `hasil_semua_model.csv` dengan kolom yang sama; prediksi CNN dapat digabung ke `preds_ml_{A,B}.parquet` untuk
  kurva ROC bersama (warna `CNN` sudah ada di `WARNA_MODEL`).
- Baseline yang harus dibandingkan dengan CNN: A AUC test ≈ 0,58; B ≈ 0,83 (SVM).
- Opsional: ML klasik dengan semua healthy train (`healthy_to_covid_ratio: null`) belum dicoba.
- Repeated CV 5×5 di train memakai hyperparameter hasil tuning pada data yang sama → sedikit optimistis (bukan
  nested CV); angka test adalah estimasi tak bias.


---

## Tahap 5 — CNN ringan pada fitur 2D (Eksperimen A)

### File yang dihasilkan
- `src/models_cnn.py` — `get_device()`/`default_batch_size()` (CUDA 64, CPU 32), `load_split()` (memmap + rata-rata waktu),
  `content_length()` (deteksi padding), `BandNormalizer` (mean/std per pita, fit di train), `SpecDataset` (augmentasi
  train: random crop di area suara, random time shift, SpecAugment), `mixup_batch()` (opsional), `LightCNN`
  (4–5 blok, kernel `3x3`/`1x12`), `CRNN` (opsional, belum dijalankan), `count_params()`, `layer_table()`,
  `predict_proba()`, `grad_cam()`.
- `src/evaluate.py` — tambah `paired_bootstrap_auc_diff()` (selisih AUC dua model pada test yang sama).
- `src/plots.py` — `plot_training_curves`, `plot_gradcam_grid`, `plot_arch_diagram`, `WARNA_SEED`; warna
  `CNN (log-mel)` / `CNN (MFCC 2D)` di `WARNA_MODEL`.
- `scripts/05_train_cnn.py` — tqdm, log ke file, resume (checkpoint `finished` dilewati; `*.last.pt` per epoch untuk
  lanjut run terputus), `--limit N` (keluaran ke `reports/logs/uji_limit/`, sudah dihapus), `--epochs`, `--seeds`,
  `--kernel`, `--n-blocks`, `--arch CRNN`, `--mixup`, `--no-specaugment`, `--tag`, `--experiment B`.
- `models/cnn_{logmel,mfcc2d}_A.pt` (seed 42) + `_seed7.pt`, `_seed2024.pt` (state dict, config, statistik normalisasi,
  ambang, riwayat, metrik val/test, prediksi); `models/preds_cnn_A.parquet` (skor val/test 6 run, kolom `cnn_<fitur>_s<seed>`).
- `reports/logs/cnn_{logmel,mfcc2d}_history.csv` (per epoch × seed), `train_cnn_A.log`, `train_cnn_A_summary.json`,
  `train_cnn_console.txt`, `nb05_exec.log`.
- `reports/tables/`: `05_cnn_per_seed.csv` (val & test per seed), `05_cnn_ringkasan.csv`, `05_overfitting.csv`,
  `05_uji_selisih_auc.csv`, `05_gradcam_padding.csv`, `05_arsitektur_lightcnn.csv`, `05_jumlah_parameter.csv`,
  `05_perbandingan_model_A.csv`; `hasil_semua_model.csv` + 2 baris (`CNN (log-mel)`, `CNN (MFCC 2D)`, seed 42 +
  kolom `cnn_seed_auc_mean/std`).
- `notebooks/05_cnn.ipynb` (dieksekusi). `reports/figures/05_*.png` (9): arsitektur, augmentasi, kurva training
  (log-mel, MFCC), perbandingan AUC, ROC/PR A, confusion A, Grad-CAM, profil frekuensi Grad-CAM.

### Protokol (keputusan)
- **Input:** fitur 2D Tahap 3, dirata-rata tiap 2 frame → resolusi 20 ms (64×500 log-mel, 40×500 MFCC). Alasan:
  CPU saja (input penuh 64×1001 ±1,2 s/batch); batuk ±200–500 ms masih terwakili.
- **Normalisasi** per pita (mean/std) di-fit pada train saja.
- **Augmentasi train saja:** random crop 256 frame (5,12 s) dengan offset acak di area berisi suara (panjang isi dideteksi
  dari frame padding yang identik dengan frame terakhir; cocok dengan `durasi_s`, r = 0,99999), shift ±20 frame,
  SpecAugment 2 freq mask (≤ 8 pita log-mel / ≤ 5 MFCC) + 2 time mask (≤ 25 frame). Mixup tersedia, tidak dipakai.
  Augmentasi level waveform (R2) tidak dikerjakan. Val/test: rekaman penuh (GAP → panjang bebas).
- **Model:** LightCNN 4 blok 3×3, 16-32-64-128, BN, MaxPool 2×2, dropout 0,10→0,20, GAP, dropout 0,30, FC→1;
  **97.521 parameter** (varian 1×12: 129.825; 5 blok: 245.233; CRNN: 245.489 — disediakan, tidak dilatih).
- **Imbalance: `pos_weight` = 3,0** (bukan WeightedRandomSampler): train sudah di-undersample 3:1; sampler akan
  mengulang ±500 rekaman COVID ±3×/epoch (risiko hafalan); pos_weight deterministik dan setara class_weight balanced ML.
- AdamW lr 1e-3, wd 1e-2; ReduceLROnPlateau (AUC val, faktor 0,5, patience 4); maks 100 epoch; early stopping AUC val
  patience 10; batch 32 (CPU, 6 thread). Ambang Youden di val; test sekali. Seed 42, 7, 2024.
- Baris utama `hasil_semua_model.csv` = seed 42 (ditetapkan sebelum melihat hasil), mean ± std di kolom tambahan.
- Uji `--limit 200 --epochs 2` (2 seed, kedua fitur) + uji resume dijalankan sebelum run penuh.

### Hasil TEST Eksperimen A (n = 1.624, prevalensi 0,067)
| Fitur | seed | epoch terbaik/total | AUC val | **AUC test [CI 95%]** | PR-AUC | Sens | Spes | menit |
|---|---|---|---|---|---|---|---|---|
| log-mel | 42 | 10/20 | 0,566 | 0,567 [0,511–0,619] | 0,081 | 0,370 | 0,708 | 8,6 |
| log-mel | 7 | 37/47 | 0,560 | 0,579 [0,525–0,631] | – | – | – | 20,3 |
| log-mel | 2024 | 9/19 | 0,567 | 0,570 [0,511–0,625] | – | – | – | 8,1 |
| MFCC 2D | 42 | 25/35 | 0,580 | 0,544 [0,487–0,601] | 0,086 | 0,463 | 0,596 | 9,2 |
| MFCC 2D | 7 | 36/46 | 0,584 | 0,591 [0,539–0,641] | – | – | – | 10,9 |
| MFCC 2D | 2024 | 43/53 | 0,595 | 0,601 [0,548–0,652] | – | – | – | 14,1 |
(Metrik lengkap per seed di `05_cnn_per_seed.csv`.)

**Mean ± std (3 seed):** log-mel AUC test **0,572 ± 0,006** (val 0,565 ± 0,004); MFCC 2D **0,579 ± 0,030**
(val 0,586 ± 0,007). PR-AUC 0,082 / 0,088. Ensemble rata-rata 3 seed (tambahan): log-mel 0,576, MFCC 0,591.
**Lama training:** total 73,3 menit untuk 6 run (CPU), 8–20 menit/run, ±25 s/epoch (log-mel), ±15 s/epoch (MFCC).

### Perbandingan dengan ML terbaik (SVM RBF, AUC test 0,576)
- Selisih AUC test (seed 42, bootstrap berpasangan 2000×): CNN log-mel − SVM = −0,009 [−0,058; 0,038], p = 0,72;
  CNN MFCC − SVM = −0,032 [−0,101; 0,040], p = 0,40; log-mel − MFCC = +0,023, p = 0,53 → **tidak ada perbedaan bermakna**.
- CNN **tidak melampaui** ML klasik; semua model A berada di AUC ≈ 0,54–0,60. Precision ≈ 0,08 → tidak layak skrining.

### Overfitting
- log-mel: ringan — selisih AUC train(teraugmentasi) − val pada checkpoint 0,02–0,06; val mendatar ±0,56 sejak ±epoch 10.
- MFCC 2D: lebih jelas — AUC train naik terus ke 0,71–0,72, val mendatar ±0,58 (selisih 0,07–0,12); early stopping menahan.
- Val loss sangat fluktuatif (hanya 108 positif, pos_weight) → AUC val lebih stabil sebagai kriteria.
- Variasi antar-seed MFCC besar (test 0,544–0,601) → seed 42 kebetulan terendah; laporkan mean ± std, bukan satu seed.

### Grad-CAM (CNN log-mel seed 42)
- Perhatian pada segmen batuk, pita ±0,2–1 kHz (puncak ±0,5 kHz) dan > 5,5 kHz; profil frekuensi **hampir identik** untuk
  COVID-19 dan healthy.
- Massa CAM di area padding ±4% vs luas padding ±57% → **tidak ada shortcut padding/durasi**.
- Skor probabilitas log-mel sangat sempit (90% test di 0,34–0,52) → keyakinan rendah; ambang Youden 0,36–0,47.

### Catatan / TODO
- CRNN dan varian kernel 1×12 sudah diimplementasi tetapi **belum dilatih** (opsional; perintah:
  `python scripts/05_train_cnn.py --features logmel --kernel 1x12 --tag _k1x12 --seeds 42` atau `--arch CRNN`).
- CNN untuk Eksperimen B didukung (`--experiment B`) tetapi tidak dijalankan (prompt tahap 5 hanya A).
- Augmentasi level waveform (noise, pitch shift, time stretch ala R2) tidak dikerjakan.
- Kalibrasi probabilitas CNN rendah (pos_weight + undersampling) → hanya pakai dengan ambang dari val.


---

## Tahap 6 — Evaluasi komprehensif, perbandingan, dan analisis (tanpa training ulang)

### File yang dihasilkan
- `src/evaluate.py` — tambah `delong_test()` (DeLong, algoritma cepat Sun & Xu; dicek vs bootstrap berpasangan) dan
  `subgroup_metrics()` (n, n_pos, AUC [CI bootstrap 500×] bila ≥ 5 pos & ≥ 5 neg, sens/spes/precision pada ambang val).
- `src/analysis.py` (baru) — `load_scores()` (gabung skor ML + CNN seed 42), `final_table()`/`fmt_final_table()`,
  `to_markdown()` (tanpa tabulate), `ml_predict()`/`cnn_predict()` (inferensi model tersimpan pada data eksperimen lain),
  `seen_uuids()` (manifest train ∪ val), `add_subgroups()` (gender, kelompok umur <25/25–34/35–44/45–54/≥55,
  respiratory_condition, fever_muscle_pain; kosong → "tidak diisi").
- `src/plots.py` — `plot_score_distributions`, `plot_error_examples`, `plot_subgroup_forest`, `plot_cross_label`.
- `scripts/06_evaluate_all.py` — langkah a–f, log `reports/logs/evaluate_all.log` + `evaluate_all_summary.json`
  (+ `evaluate_all_console.txt`), opsi `--skip-cnn`. Run penuh ±80 s (CPU). Tidak ada `--limit` (tidak ada proses panjang).
- `notebooks/06_evaluasi_akhir.ipynb` (dieksekusi; menampilkan tabel/figur + contoh `delong_test`), log `nb06_exec.log`.
- `reports/tables/06_*`: `tabel_final_{A,B}` (.csv numerik, `_format.csv`, .md), `uji_signifikansi_auc` (.csv/.md),
  `kesepakatan_label_B` (.csv/.md), `uji_silang_label` (.csv/.md), `distribusi_skor_A`, `contoh_fp_fn_A`,
  `karakteristik_error_A` (.csv/.md), `korelasi_skor_durasi_A`, `subgrup_fairness` (.csv long, .md),
  `subgrup_selisih_auc`, `perbandingan_literatur` (.csv/.md).
- `reports/figures/06_*.png` (6): `roc_gabungan_A`, `uji_silang_label`, `distribusi_skor_A`, `contoh_fp_fn_A`,
  `subgrup_svm_A`, `subgrup_cnn_logmel_A`.
- `reports/ringkasan_hasil.md` (bahan utama Tahap 7), `reports/daftar_figur_tabel.md` (Gambar 1–41, Tabel 1–18,
  kolom "Inti" = saran masuk badan laporan).

### Keputusan
- "Model terbaik" ditandai dua cara: ★ = AUC val tertinggi (kriteria seleksi yang ditetapkan sebelumnya) dan ▲ = AUC
  test tertinggi (informasi). Untuk A: ★ = CNN (MFCC 2D) seed 42 (val 0,580, test 0,544), ▲ = k-NN (0,580).
  ML terbaik menurut val tetap SVM (RBF) → dipakai sebagai "ML terbaik" di ROC gabungan, analisis error & subgrup.
- Uji signifikansi: dua model teratas menurut AUC test + pasangan kunci (SVM vs CNN, SVM vs dummy; B: dua teratas, vs dummy).
- Uji silang: dari set evaluasi dibuang uuid **manifest train ∪ val** model sumber (yang benar-benar dilihat model;
  rekaman healthy yang terbuang undersampling tidak pernah dipakai). Versi pertama memakai populasi split penuh
  (S1 hanya 79 rekaman) → diganti.
  S1 = test B tanpa uuid A (n = 102, 51 COVID ahli); S2 = test A tanpa uuid B (n = 1.554, 79 COVID);
  S3 = semua rekaman B yang tak dilihat model A dan punya status biner (n = 251) → prediksi identik dinilai dengan dua label.
  CNN A ikut diuji di S1/S3 (normalisasi dari statistik train A di checkpoint). CNN B tidak ada (tidak dilatih).
- Contoh FP/FN = 3 kesalahan paling yakin SVM (skor tertinggi pada healthy / terendah pada COVID-19).
- Tabel literatur: angka diekstrak dari PDF `references/` (pdftotext, dengan nomor halaman). R0 tidak ada PDF-nya → "lihat paper".

### Hasil utama
- **Tabel final A (test):** semua AUC 0,544–0,580; PR-AUC 0,081–0,097 (prevalensi 0,067); precision 0,07–0,09.
- **Signifikansi (DeLong / bootstrap):** k-NN − SVM +0,004 (p 0,87); SVM − CNN log-mel +0,009 (p 0,70);
  SVM − CNN MFCC +0,032 (p 0,39); **SVM − dummy +0,060 (p 0,12, tidak bermakna)**. B: SVM − RF +0,009 (p 0,62);
  SVM − dummy +0,339 (p < 0,001).
- **Kesepakatan label** (manifest B, 572 rekaman dengan status biner): 50,3%, **κ = 0,00**; hanya 43% yang mengaku
  COVID-19 dinilai COVID-19 oleh ahli.
- **Uji silang (SVM):** S1: model B 0,896 vs model A 0,673 (ML A 0,61–0,71; CNN log-mel 0,653; CNN MFCC 0,489);
  S2: model A 0,585 vs model B 0,545 (B semua 0,55–0,58); S3: model A vs label ahli 0,574, vs self-reported 0,541.
  Catatan: label ahli = penilaian dari mendengar audio yang sama (bukan PCR) → "konsisten akustik" secara konstruksi.
- **Error:** skor sempit (SVM median healthy 0,242 vs COVID 0,254). FP: median durasi 5,0 s (TN 3,0 s); FN: banyak
  rekaman sangat pendek (0,8–1,0 s). Spearman skor–durasi: SVM 0,41, CNN log-mel 0,61, CNN MFCC −0,05; durasi saja AUC 0,518.
- **Subgrup (SVM A):** demam ya 0,507 vs tidak 0,590; kondisi napas ya 0,529 vs tidak 0,586; umur ≥ 55 0,475,
  35–44 0,631; female 0,585, male 0,565. Prevalensi COVID-19 di subgrup demam 0,23 vs 0,05 → gejala *confounder*.
  Semua CI subgrup lebar (12–73 positif).
- **Literatur:** R2/R3 memakai Cambridge COVID-19 Sounds (multimodal; accuracy/F1 91–95%; tanpa AUC; tidak per subjek);
  R4 memakai data call-centre MedInGroup (RF AUC 0,83 PCR vs 0,79 self-reported). Tidak ada yang memakai COUGHVID.

### TODO (cek manual oleh pengguna)
- `06_perbandingan_literatur.md`: semua sel bertanda **[TODO: cek manual]** —
  R0 (PDF tidak ada; isi "lihat paper"), R2 (sumber label; label tugas Tabel 2–5 tidak konsisten),
  R3 (1.541 file vs ±13.000 sampel), R4 (Tabel 2 vs kalimat "selisih 0,02–0,10" di hlm. 4).
- Opsional: Eksperimen C (`status_SSL`), kalibrasi probabilitas, fitur per segmen batuk, CNN untuk B.



---

## Tahap 7a — Draf laporan per bab (Markdown) — `reports/final/draft/`

### Status bab
| File | Status |
|---|---|
| `reports/figures/07_alur_penelitian.png` (diagram, `scripts/07_diagram_alur.py`) | ✅ |
| `01_pendahuluan.md` | ✅ |
| `02_tinjauan_pustaka.md` | ✅ |
| `03_metodologi.md` | ✅ |
| `04_hasil_dan_pembahasan.md` | ✅ |
| `05_kesimpulan.md` | ✅ |
| `00_abstrak.md` (ID 182 kata + EN 208 kata) | ✅ |
| `06_daftar_pustaka.md` (IEEE, 38 referensi, dibuat otomatis) | ✅ |
| `07_lampiran.md` | ✅ |

### File yang dihasilkan
- `reports/final/draft/00_abstrak.md … 07_lampiran.md` — **versi bernomor** (sitasi [1], [2], …; siap untuk Tahap 7b).
- `reports/final/draft/_kunci/*.md` — **sumber berkunci** (`[@kunci]`). **Edit di sini**, lalu jalankan
  `python scripts/07_sitasi.py` → menomori ulang sitasi (urutan kemunculan pertama) + membuat ulang `06_daftar_pustaka.md`.
- `scripts/07_sitasi.py` (dict `IEEE` = entri daftar pustaka), `scripts/07_diagram_alur.py` +
  `src/plots.py::plot_research_flow()` → `reports/figures/07_alur_penelitian.png`.
- `references/references.bib`: penulis lengkap R4 diperbaiki (dari PDF).

### Konvensi draf (untuk Tahap 7b)
- Penomoran laporan **berbeda** dari `daftar_figur_tabel.md`: badan laporan memakai Gambar 1–23 dan Tabel 1–19
  (Bab II–IV), lampiran Tabel L1–L4 + daftar gambar tambahan (Lampiran D, tanpa nomor).
- Gambar: baris `![alt](../../figures/xxx.png)` diikuti baris caption tebal `**Gambar n.** …` (caption di bawah).
  Tabel: caption tebal `**Tabel n.** …` di atas tabel Markdown; komentar `<!-- sumber: reports/tables/... -->` di
  bawah tiap tabel menunjuk CSV asal (untuk render ulang dari CSV di 7b).
- Rumus dalam LaTeX (`$…$`, `$$…$$`) → perlu pandoc (atau konversi OMML) di 7b.
- Angka di teks/tabel memakai koma desimal (gaya Indonesia), 3 desimal sesuai CSV.

### Keputusan
- Angka hasil hanya dari `reports/tables/` dan PROGRESS (yang bersumber dari tabel/log). Angka literatur dari PDF
  `references/` (R1 untuk Laguarta: **akurasi 97,1%**, bukan AUC).
- Referensi tambahan (teori, algoritma, uji statistik, pustaka librosa/scikit-learn/PyTorch/imblearn/XGBoost/spafe/
  noisereduce) ditambahkan dari pengetahuan umum → **verifikasi detail bibliografinya** (volume/halaman).
- Pembagian tugas di Lampiran E adalah usulan (3 anggota) dengan placeholder `[NAMA ANGGOTA]` dan `[NIM]`.

### TODO / cek manual oleh pengguna
1. Isi `[NAMA ANGGOTA]` dan `[NIM]` (Lampiran E); sesuaikan pembagian tugas/jumlah anggota.
2. Referensi `sela2026materi` (materi kuliah): lengkapi nama institusi/tahun bila perlu.
3. Verifikasi entri IEEE non-R0–R4 di `scripts/07_sitasi.py` (terutama spafe/JOSS 2023, noisereduce/Sainburg 2020).
4. Angka literatur yang di Tahap 6 bertanda [TODO: cek manual] dipakai di Tabel 1 & 19: R2 (label & akurasi
   batuk saja 93,12%), R3 (jumlah data), R4 (RF 0,83 vs 0,79). R0 tidak ada PDF-nya (hanya fakta dari data kita + R1).
5. Tahap 7b: konversi ke .docx (sampul, TOC, daftar gambar/tabel, render tabel dari CSV).


---

## Tahap 7b — Konversi ke dokumen akhir (Word + PDF)

### Yang dikerjakan
- `scripts/07b_buat_docx.py` (jalankan: `python scripts/07b_buat_docx.py`; opsi `--no-pdf`, `--no-word`):
  1. Membuat `reports/final/reference.docx` dari reference bawaan pandoc → Times New Roman 12, spasi 1,5,
     margin 3-3-3-3 cm, A4, Heading 1 rata tengah + ganti halaman; gaya tambahan `Keterangan Gambar`,
     `Keterangan Tabel`, `Isi Tabel`, `Daftar Pustaka` (hanging indent), `Sampul *`, `Judul Depan`.
  2. Menggabungkan `draft/00…07*.md` + sampul + Daftar Isi/Gambar/Tabel (field TOC) → `reports/final/_gabungan.md`
     → pandoc 3.9 (`pypandoc_binary`; pandoc sistem tidak terpasang). Rumus LaTeX → persamaan Word (OMML).
  3. Pasca-proses python-docx: caption gambar (di bawah) & tabel (di atas, *keep with next*), gambar lebar ≤ 15 cm
     (tinggi ≤ 20 cm), tabel bergaris + header abu-abu berulang + lebar 100% + 10 pt, 3 seksi
     (sampul tanpa nomor; abstrak–daftar tabel romawi i–x; Bab I–lampiran arab mulai 1).
  4. MS Word via COM (pywin32): perbarui semua field/TOC lalu ekspor PDF (dengan *bookmark* heading).

### Hasil
- `reports/final/Laporan_COUGHVID_Kelompok.docx` (±6,2 MB) dan `.pdf` (±3,1 MB) — **64 halaman**
  (10 halaman depan + 54 isi).
- 23/23 gambar muncul; 23 tabel (Tabel 1–19 + L1–L4) sebagai tabel Word; 38 entri pustaka; daftar isi 80 entri,
  daftar gambar 23, daftar tabel 23. Placeholder tersisa hanya `[NAMA ANGGOTA]` dan `[NIM]` (sampul & Tabel L4).

### Keputusan / catatan teknis
- Tabel dirender dari tabel Markdown di draf (isinya sudah diambil dari CSV `reports/tables/` di Tahap 7a,
  3 desimal, koma desimal) — tidak dibaca ulang dari CSV agar subset kolom & format di draf tetap.
- Switch `\t "Gaya,1"` pada field daftar gambar/tabel harus memakai pemisah daftar locale; locale Indonesia = `;`.
  Script membaca `word.International(wdListSeparator)` dan menyesuaikan kode field otomatis.
- Judul di `00_abstrak.md` tidak dipakai (sudah di sampul); kalimat "Gaya IEEE; …" di daftar pustaka dihapus.
- Nama universitas/logo tidak dicantumkan di sampul (tidak ada di instruksi).
- Bila draf diubah: edit `draft/_kunci/*.md` → `python scripts/07_sitasi.py` → `python scripts/07b_buat_docx.py`.

### TODO / cek manual oleh pengguna
1. Isi `[NAMA ANGGOTA]` dan `[NIM]` di sampul (3 baris) dan Tabel L4 (Lampiran E); sesuaikan jumlah anggota.
   Lebih aman diisi di draf (`_kunci/07_lampiran.md`, dan `sampul()` di `scripts/07b_buat_docx.py`) lalu build ulang.
2. Tambahkan nama universitas/logo di sampul bila diwajibkan.
3. Butir TODO Tahap 7a masih berlaku (verifikasi entri pustaka non-R0–R4, angka literatur bertanda cek manual).
4. Jika dokumen diedit manual di Word: klik kanan daftar isi/gambar/tabel → *Update Field* → *Update entire table*.


---

## Tahap 8 — Slide presentasi, demo, README, QA, dan paket pengumpulan

### File yang dihasilkan
- `scripts/08_buat_slide.py` → `reports/final/Presentasi_COUGHVID_Kelompok.pptx` (python-pptx, 16:9, **15 slide**,
  Cambria/Calibri, navy `#102A3C` + oranye/biru = warna kelas COVID-19/healthy di figur). Motif: label tahap
  pipeline (pill) di tiap slide konten + ilustrasi "gelombang batuk" di slide judul & penutup.
  Urutan: judul & anggota → latar belakang → rumusan masalah & tujuan → dataset → alur (07_alur_penelitian) →
  preprocessing (before/after) → representasi suara → fitur & uji statistik → model → hasil A (tabel dari
  `06_tabel_final_A.csv` + ROC) → Eksp. B / keandalan label → analisis error & subgrup → literatur →
  kesimpulan & saran → terima kasih/Q&A. **Speaker notes** Bahasa Indonesia tiap slide (97–127 kata ≈ 50–65 s pada
  ±115 kata/menit; dengan jeda dan menunjuk grafik ±60–90 s). Lolos `validate.py` skill pptx; dicek visual (render
  PowerPoint) — teks tidak terpotong/bertumpuk.
- `scripts/predict.py` — demo inferensi 1+ file audio (format apa pun yang didukung ffmpeg) → preprocessing a–e +
  315 fitur yang sama → P(COVID-19) + label (ambang Youden dari val). Default **SVM Eksperimen A** (ML terbaik menurut
  AUC val); opsi `--eksperimen B`, `--algo`, `--semua`, `--json`. Disclaimer "bukan alat diagnosis medis" di awal
  keluaran. File sementara di temp sistem (tidak menulis ke `data/`). Audio senyap/terlalu pendek ditolak (exit 1).
  **Diverifikasi**: 6 rekaman test (3 COVID, 3 healthy) dari file mentah → skor SVM & RF identik dengan skor test
  tersimpan (selisih maks 4,7·10⁻⁵ = pembulatan). `app_demo.py` (Gradio) **tidak dibuat** — gradio tidak terpasang
  dan opsional.
- `scripts/08_qa_final.py` → `reports/logs/qa_final.md` — **158/158 pemeriksaan lulus** (leakage + konsistensi angka).
- `README.md` (deskripsi, hasil, struktur, instalasi, urutan 01→06, reproduksi, demo, lisensi CC BY 4.0 + sitasi).
- `requirements-lock.txt` (`pip freeze`, 154 paket + baris `--extra-index-url` PyTorch CPU agar bisa langsung diinstal).
- `submission/`: `Laporan_COUGHVID_Kelompok.docx`, `.pdf`, `Presentasi_COUGHVID_Kelompok.pptx`,
  `Kode_COUGHVID_Kelompok.zip` (5,4 MB, 35 file: src, scripts, notebooks, config.yaml, README, requirements,
  requirements-lock, references.bib, ringkasan_hasil.md, qa_final.md; **tanpa** data/, models/, .venv).
  `submission/` dimasukkan ke `.gitignore` (salinan hasil build).

### QA
1. **Notebook 01–06 dieksekusi ulang** (`nbconvert --execute --inplace`, kernel ppkb): 0 error, total ±3,2 menit
   (log `reports/logs/nb_final_exec.log`). Semua tabel `reports/tables/` dan figur `reports/figures/` yang ditulis ulang
   **identik byte-per-byte** dengan versi commit (git tidak mendeteksi perubahan) → hasil reprodusibel.
2. **Leakage** (otomatis): uuid train/val/test saling lepas (splits & manifest A/B); val/test tidak di-undersample
   (prevalensi alami); fitur tabular & array 2D = manifest per split; 12 pipeline ML: scaler → SelectKBest → [SMOTE] →
   clf di dalam satu pipeline imblearn (di-fit hanya pada train; SMOTE tidak berlaku saat predict); ambang dari val;
   normalisasi CNN di checkpoint = statistik train saja (selisih 0 vs train; 0,06–0,12 vs train+val); augmentasi hanya
   `SpecDataset(train=True)`; uji silang S1/S2 tanpa uuid train/val model sumber. Code review `04_train_ml.py` /
   `05_train_cnn.py` mengonfirmasi tuning CV hanya di train.
3. **Konsistensi angka**: 111 angka kunci dari CSV (tabel final A/B semua model, CI, PR-AUC, sens/spes, p DeLong,
   kesepakatan label, uji silang, median skor, Spearman, subgrup, jumlah fitur, parameter, populasi) ditemukan di
   laporan .docx dan/atau slide sesuai harapan. Slide tabel hasil dibaca langsung dari CSV.
   Minor: `06_perbandingan_literatur.csv` menyebut B = 1.013 rekaman (sebelum preprocessing; 2 dibuang) — laporan & slide
   memakai 1.011 (470/541) yang benar; CSV dibiarkan.

### Checklist pengumpulan
- [x] Laporan `.docx` + `.pdf` (64 halaman) — `submission/`
- [x] Slide `.pptx` (15 slide + speaker notes) — `submission/`
- [x] Zip kode tanpa data/.venv — `submission/Kode_COUGHVID_Kelompok.zip`
- [x] README, requirements.txt, requirements-lock.txt
- [x] Notebook dieksekusi ulang tanpa error; QA leakage & angka lulus
- [ ] Isi `[NAMA ANGGOTA]` / `[NIM]`: sampul laporan + Tabel L4 (lewat `_kunci/07_lampiran.md` & `sampul()` di
      `scripts/07b_buat_docx.py` → build ulang), slide 1 (`scripts/08_buat_slide.py`), tabel anggota di README;
      nama kelompok di speaker notes slide 1 (`[nama kelompok]`)
- [ ] Nama universitas/logo di sampul & slide judul bila diwajibkan
- [ ] Verifikasi entri pustaka non-R0–R4 dan angka literatur bertanda [TODO: cek manual] (R2 93,12%, R3 jumlah data,
      R4 0,83 vs 0,79) — dipakai juga di slide 2 & 13
- [ ] Setelah mengedit: `python scripts/07b_buat_docx.py`, `python scripts/08_buat_slide.py`,
      `python scripts/08_qa_final.py`, lalu salin ulang ke `submission/` (dan buat ulang zip)
- [ ] Latihan presentasi dengan speaker notes (target ±15 menit) dan uji demo `predict.py` di laptop presentasi
