# PROGRESS — Proyek PPKB (COUGHVID)

## Status tahap
| Tahap | Status | Tanggal |
|---|---|---|
| 1. Setup, ekstraksi, eksplorasi metadata | ✅ Selesai | 2026-11-26 |
| 2. Preprocessing audio | ⏳ Belum | |

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
