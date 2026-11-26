# CLAUDE.md — Konteks Proyek "PPKB" (COUGHVID)

> File ini diletakkan di root proyek dengan nama **`CLAUDE.md`**. Claude Code membacanya otomatis
> di setiap sesi, jadi konteks inti tidak perlu diulang di setiap prompt.

## 1. Latar belakang tugas
- Mata kuliah: **Pengenalan Pola dan Kecerdasan Buatan** (S2 Magister Teknologi Informasi).
- Materi: Pertemuan 8 — Pengenalan Pola Suara (dosen: Enny Itje Sela).
- Tugas (slide 2): **membuat kasus yang melibatkan pengenalan suara**, berkelompok 2–3 mahasiswa.
- Kasus kelompok: **Klasifikasi suara batuk untuk skrining COVID-19 menggunakan dataset COUGHVID v3**
  (Orlandic, Teijeiro & Atienza, Zenodo record 7024894).
- Alur kerja WAJIB mengikuti pipeline di slide 10 materi:
  1. **Akuisisi** (dataset crowdsourced, rekaman mikrofon smartphone/web)
  2. **Preprocessing** (konversi format, resampling, trimming/segmentasi, normalisasi, reduksi noise)
  3. **Feature extraction** (energy, ZCR, MFCC, + GFCC / spektral / log-mel spectrogram)
  4. **Classification** (model ML klasik + CNN ringan)
  5. **Evaluation** (performa pada data uji)
- Konsep dari materi yang harus tampak di laporan: amplitudo, frekuensi, periode, pitch, timbre,
  suara periodik vs nonperiodik (batuk = dominan nonperiodik), analog vs digital, sampling &
  quantization (sampling rate, bit depth), waveform, domain waktu vs frekuensi, Fourier/FFT,
  spectrogram.

## 2. Lingkungan
- OS: **Windows**, shell: PowerShell. Nama file dataset mengandung spasi → **selalu beri tanda kutip** pada path.
- Root proyek: `D:\Meidhika\Lecturer\project-kuliah\PPKB`
- Dataset zip: `<root>\The COUGHVID crowdsourcing dataset.zip` (±2.3 GB). **Jangan dihapus/diubah.**
- Python 3.10/3.11 dalam virtual env `.venv` di root proyek. Deep learning memakai **PyTorch**
  (CPU harus tetap jalan; pakai GPU/CUDA bila tersedia).
- ffmpeg diperlukan untuk dekode `.webm`/`.ogg`.

## 3. Struktur folder (target)
```
PPKB/
├── CLAUDE.md               ← file ini
├── PROGRESS.md             ← log serah-terima antar tahap (WAJIB diperbarui di akhir tiap tahap)
├── README.md
├── requirements.txt
├── config.yaml             ← semua parameter (sr, durasi, n_mfcc, seed, path, threshold, dst.)
├── The COUGHVID crowdsourcing dataset.zip
├── prompts/                ← file prompt tahap (untuk pengguna; JANGAN dibaca/diubah kecuali diminta)
├── data/
│   ├── raw/                ← hasil ekstrak zip (read-only setelah diekstrak)
│   ├── interim/            ← wav 16 kHz mono hasil konversi
│   ├── processed/          ← wav tersegmentasi/ternormalisasi + manifest CSV
│   └── features/           ← .parquet/.npy fitur
├── src/                    ← SEMUA logika ada di sini (modul .py yang bisa di-import)
│   ├── config.py  data_utils.py  preprocessing.py  features.py
│   ├── models_ml.py  models_cnn.py  evaluate.py  plots.py
├── scripts/                ← entry point CLI: 01_extract.py, 02_preprocess.py, ...
├── notebooks/              ← notebook naratif per tahap (tipis, memanggil src/)
├── models/                 ← model tersimpan (.joblib / .pt)
├── reports/
│   ├── figures/            ← semua gambar PNG (dpi 200) untuk laporan
│   ├── tables/             ← tabel hasil CSV/Markdown
│   └── final/              ← laporan .docx/.pdf + slide presentasi
└── references/             ← PDF artikel referensi + references.bib
```

## 4. Aturan kerja (penting untuk menjaga context window)
1. **Kerjakan hanya tahap yang diminta prompt.** Jangan melompat ke tahap berikutnya.
2. Di awal sesi: baca `CLAUDE.md` dan `PROGRESS.md` saja. Jangan membaca ulang seluruh kode
   kecuali perlu; gunakan `grep`/baca bagian yang relevan.
3. **Jangan pernah mencetak output besar** ke terminal (list ribuan file, isi CSV penuh, array).
   Gunakan `head`, ringkasan, `value_counts()`, atau tulis ke file log di `reports/logs/`.
4. Proses panjang (konversi audio, ekstraksi fitur, training) dijalankan sebagai script dengan
   **progress bar (tqdm)**, **logging ke file**, **bisa dilanjutkan (resume/skip file yang sudah ada)**,
   dan opsi `--limit N` untuk uji cepat. Selalu uji dulu dengan `--limit 50` sebelum run penuh.
5. Logika di `src/`, notebook hanya memanggil fungsi + menampilkan grafik. Notebook dieksekusi
   dengan `jupyter nbconvert --to notebook --execute --inplace` agar output tersimpan.
6. Reproducibility: `seed = 42` di semua tempat (numpy, random, torch, sklearn).
7. **Anti data leakage:** split train/val/test dilakukan SEBELUM oversampling, augmentasi,
   scaling, dan seleksi fitur. Scaler/SMOTE/augmentasi hanya di-fit pada train.
8. Kejujuran ilmiah: laporkan angka apa adanya. COUGHVID berlabel *self-reported* dan noisy;
   AUC 0.60–0.75 itu wajar. **Jangan** mengubah eksperimen agar angka terlihat seperti paper
   (beberapa paper melaporkan >90% dengan setting yang berbeda/berisiko leakage).
9. Bahasa: komentar kode, notebook, dan laporan dalam **Bahasa Indonesia** (istilah teknis
   boleh bahasa Inggris dengan huruf miring).
10. Di akhir setiap tahap: perbarui `PROGRESS.md` (apa yang selesai, file yang dihasilkan,
    angka penting, keputusan yang diambil, masalah/TODO), lalu berhenti dan beri ringkasan singkat.

## 5. Fakta dataset COUGHVID v3 (verifikasi dengan data asli — jangan diasumsikan)
- >30.000 rekaman batuk crowdsourced (s.d. Okt 2021), format `.webm`/`.ogg` + `.json` per rekaman.
- Metadata utama: `metadata_compiled.csv`. Kolom yang biasanya ada: `uuid`, `datetime`,
  `cough_detected` (probabilitas dari classifier batuk; rekomendasi penulis dataset: ≥ 0.8),
  `latitude`, `longitude`, `age`, `gender`, `respiratory_condition`, `fever_muscle_pain`,
  `status` (*self-reported*: healthy / symptomatic / COVID-19), `status_SSL` (label hasil
  *semi-supervised learning*, khusus v3), serta label ahli untuk ±2.000 rekaman dari hingga 4
  pulmonolog: `quality_1..4`, `cough_type_1..4`, `dyspnea_*`, `wheezing_*`, `stridor_*`,
  `choking_*`, `congestion_*`, `nothing_*`, `diagnosis_1..4`, `severity_1..4`.
- Kemungkinan ada zip bersarang (`public_dataset_v3.zip` di dalam zip unduhan). Cek dulu.
- Tidak ada ID subjek → split per rekaman (`uuid`). Catat sebagai keterbatasan.

## 6. Ringkasan referensi (agar tidak perlu membaca ulang PDF)
| # | Referensi | Inti yang dipakai |
|---|-----------|-------------------|
| R1 | Lella & Alphonse (2021), *A literature review on COVID-19 disease diagnosis from respiratory sound data*, AIMS Bioengineering 8(2):140–153, doi:10.3934/bioeng.2021013 | Tinjauan dataset crowdsourced (COUGHVID, Coswara, COVID-19 Sounds) dan metode (MFCC, CNN, SVM); dasar tinjauan pustaka |
| R2 | Lella & Pja (2022), *Automatic diagnosis of COVID-19 disease using deep CNN with multi-feature channel from respiratory sound data: cough, voice, and breath*, Alexandria Eng. J. 61:1319–1334, doi:10.1016/j.aej.2021.06.024 | Multi-feature channel (De-noising Autoencoder, GFCC, IMFCC) + augmentasi (time stretch, pitch shift, noise) → DCNN, Softmax |
| R3 | Kranthi Kumar & Alphonse (2022), *COVID-19 disease diagnosis with light-weight CNN using modified MFCC and enhanced GFCC from human respiratory sounds*, Eur. Phys. J. Spec. Top. 231:3329–3346, doi:10.1140/epjs/s11734-022-00432-w | CNN ringan, M-MFCC & E-GFCC, receptive field 1×12, kedalaman 5 layer, batch norm, 100 epoch, categorical cross-entropy |
| R4 | Xiong et al. (2022), *Reliability of crowdsourced data and patient-reported outcome measures in cough-based COVID-19 screening*, Sci. Rep. 12:21990, doi:10.1038/s41598-022-26492-5 | Label self-reported kurang andal vs PCR; openSMILE (6.373 fitur) + seleksi fitur; RF terbaik (AUC ≈ 0.8); gunakan AUC karena data imbalanced |
| R0 | Orlandic, Teijeiro & Atienza (2021), *The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms*, Scientific Data 8:156, doi:10.1038/s41597-021-00937-4 | Sumber dataset (wajib disitasi) |

## 7. Label eksperimen (ditetapkan di Tahap 2)
- **Eksperimen utama (A):** `status` = COVID-19 (1) vs healthy (0), filter `cough_detected ≥ 0.8`.
- **Eksperimen B (keandalan label, terinspirasi R4):** subset berlabel ahli —
  `diagnosis_*` = COVID-19 vs healthy_cough — dibandingkan dengan label self-reported.
- **Eksperimen C (opsional):** `status_SSL` sebagai label.
