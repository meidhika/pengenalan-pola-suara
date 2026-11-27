# Klasifikasi Suara Batuk untuk Skrining COVID-19 — COUGHVID v3

Tugas kelompok mata kuliah **Pengenalan Pola dan Kecerdasan Buatan** (Magister Teknologi Informasi,
Pertemuan 8 — Pengenalan Pola Suara; dosen: Enny Itje Sela).

Proyek ini membangun *pipeline* pengenalan pola suara batuk mengikuti lima tahap materi
(**akuisisi → preprocessing → ekstraksi fitur → klasifikasi → evaluasi**) pada dataset
[COUGHVID v3](https://doi.org/10.5281/zenodo.7024894), lalu membandingkan *machine learning* klasik
(fitur MFCC, GFCC, spektral, domain waktu) dengan CNN ringan (log-mel dan MFCC 2D). Semua pembagian data
dilakukan **sebelum** *undersampling*, SMOTE, augmentasi, *scaling*, dan seleksi fitur (anti *data leakage*), dan
`seed = 42` dipakai di semua tempat.

> ⚠️ **Bukan alat diagnosis medis.** Model dan demo di repositori ini dibuat untuk pembelajaran. Hasilnya tidak
> boleh dipakai untuk menyimpulkan seseorang positif atau negatif COVID-19.

## Ringkasan hasil

Semua angka berasal dari split **test** yang dievaluasi sekali, dengan ambang Youden yang dipilih di data validasi.
Sumber: `reports/tables/06_tabel_final_{A,B}.csv`, `reports/ringkasan_hasil.md`.

| Setting | Model | ROC-AUC test [CI 95%] | PR-AUC | Sens. / Spes. |
|---|---|---|---|---|
| **A** — label *self-reported*, n = 1.624 (108 COVID-19, prevalensi 0,067) | SVM RBF | 0,576 [0,523–0,629] | 0,088 | 0,556 / 0,553 |
| A | k-NN (AUC test tertinggi) | 0,580 [0,525–0,636] | 0,088 | 0,509 / 0,613 |
| A | CNN log-mel (mean 3 seed) | 0,572 ± 0,006 | 0,081 | 0,370 / 0,708 (seed 42) |
| A | CNN MFCC 2D (mean 3 seed) | 0,579 ± 0,030 | 0,086 | 0,463 / 0,596 (seed 42) |
| A | Dummy (baseline acak) | 0,516 [0,472–0,563] | 0,069 | – |
| **B** — label diagnosis pulmonolog, n = 152 | SVM RBF + SMOTE | 0,826 [0,745–0,888] | 0,786 | 0,743 / 0,805 |

Temuan utama:

1. Pada label *self-reported* semua model hanya mencapai ROC-AUC 0,54–0,58. Selisih antarmodel tidak bermakna
   (DeLong p ≥ 0,39), dan SVM vs *dummy* juga tidak bermakna (p = 0,12). CNN ringan tidak mengungguli ML klasik.
   Precision ≈ 0,08, sehingga model **tidak layak untuk skrining**.
2. Dengan fitur dan *pipeline* yang sama, label ahli memberi AUC 0,826. Kesepakatan label *self-reported* dan label
   ahli pada 572 rekaman hanya 50,3% (Cohen's κ = 0,00). Kualitas label adalah faktor pembatas utama, sejalan dengan
   Xiong dkk. (2022). Catatan: label ahli berasal dari penilaian pendengaran pulmonolog, bukan PCR.
3. Skor model berkorelasi dengan durasi/kepadatan batuk (Spearman ρ 0,41 untuk SVM), dan kinerjanya lebih rendah pada
   subgrup bergejala (demam: AUC 0,51 vs 0,59).

## Struktur folder

```
PPKB/
├── README.md, CLAUDE.md, PROGRESS.md   ← dokumentasi, konteks proyek, log per tahap
├── config.yaml                         ← SEMUA parameter (sr, durasi, fitur, split, seed, path)
├── requirements.txt                    ← dependensi (rentang versi)
├── requirements-lock.txt               ← versi persis lingkungan yang dipakai (pip freeze)
├── The COUGHVID crowdsourcing dataset.zip   ← unduhan Zenodo (tidak disertakan di repo/zip)
├── src/                  ← logika (modul yang di-import notebook & script)
│   ├── config.py           loader config + set_seed
│   ├── data_utils.py       metadata, label A/B/C, split, undersampling
│   ├── preprocessing.py    konversi ffmpeg, trim, segmentasi, reduksi noise, normalisasi
│   ├── features.py         fitur tabular (315) & 2D (log-mel, MFCC-2D), uji Mann-Whitney + FDR
│   ├── models_ml.py        pipeline Scaler → SelectKBest → [SMOTE] → klasifikator
│   ├── models_cnn.py       LightCNN, dataset + augmentasi, normalisasi per pita, Grad-CAM
│   ├── evaluate.py         metrik, CI bootstrap, ambang Youden, DeLong
│   ├── analysis.py         tabel final, uji silang label, subgrup
│   └── plots.py            semua figur
├── scripts/              ← entry point CLI (urutan 01 → 08)
├── notebooks/            ← notebook naratif per tahap (01–06, sudah dieksekusi)
├── data/                 ← dibuat oleh script (tidak disertakan; hanya splits_*.csv yang di-commit)
│   ├── raw/  interim/  processed/  features/
├── models/               ← model tersimpan .joblib / .pt (dibuat oleh script 04–05)
├── reports/
│   ├── figures/  tables/  logs/
│   ├── ringkasan_hasil.md, daftar_figur_tabel.md
│   └── final/            ← laporan .docx/.pdf, slide .pptx, draf Markdown
└── references/           ← PDF referensi + references.bib
```

## Instalasi (Windows / PowerShell)

Prasyarat: **Python 3.11** (3.10 juga bisa) dan **ffmpeg** di PATH (untuk dekode `.webm`/`.ogg`/AAC). Tidak perlu
GPU; seluruh proyek ini dijalankan di CPU.

```powershell
cd D:\Meidhika\Lecturer\project-kuliah\PPKB
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # PyTorch CPU
pip install -r requirements.txt
# reproduksi versi persis: pip install -r requirements-lock.txt
python -m ipykernel install --user --name ppkb --display-name "Python (ppkb)"
ffmpeg -version   # pastikan ffmpeg terpasang
```

Letakkan file unduhan Zenodo `The COUGHVID crowdsourcing dataset.zip` di root proyek (nama file mengandung spasi,
jadi selalu beri tanda kutip pada path).

## Menjalankan pipeline (urutan 01 → 06)

Setiap script mencatat log ke `reports/logs/`, menampilkan *progress bar*, bisa dilanjutkan (*resume*, file yang
sudah ada dilewati), dan menyediakan `--limit N` untuk uji cepat. Jalankan dulu dengan `--limit 50`.

| # | Perintah | Hasil | Waktu (CPU 6 core) |
|---|---|---|---|
| 1 | `python scripts/01_extract.py` | `data/raw/coughvid_20211012/` (34.434 audio + JSON + metadata) | beberapa menit (ekstrak ±2,3 GB) |
| 2 | `python scripts/02_preprocess.py` | split (`data/processed/splits_{A,B}.csv`), wav 16 kHz hasil preprocessing, `manifest_{A,B}.csv` | ±5 menit |
| 3 | `python scripts/03_extract_features.py` | `data/features/tabular_{A,B}.parquet`, array log-mel & MFCC-2D | ±90 menit (pYIN) |
| 4 | `python scripts/04_train_ml.py` | `models/ml_*_{A,B}.joblib`, `reports/tables/hasil_semua_model.csv`, figur 04 | ±20 menit |
| 5 | `python scripts/05_train_cnn.py` | `models/cnn_{logmel,mfcc2d}_A*.pt` (3 seed), figur/tabel 05 | ±75 menit |
| 6 | `python scripts/06_evaluate_all.py` | tabel final, uji DeLong, uji silang label, analisis error & subgrup (`reports/tables/06_*`) | ±1,5 menit |

Notebook `notebooks/01…06` menampilkan narasi, tabel, dan grafik tiap tahap. Notebook memanggil fungsi `src/` dan
memuat hasil script (tidak melatih ulang). Jalankan ulang semuanya dengan:

```powershell
Get-ChildItem notebooks\0*.ipynb | ForEach-Object { jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=ppkb $_.FullName }
```

Dokumen akhir (opsional, butuh MS Word untuk TOC & PDF):

```powershell
python scripts/07_diagram_alur.py      # diagram alur penelitian
python scripts/07_sitasi.py            # penomoran sitasi IEEE draf laporan
python scripts/07b_buat_docx.py        # laporan .docx + .pdf
python scripts/08_buat_slide.py        # slide presentasi .pptx (15 slide + speaker notes)
python scripts/08_qa_final.py          # QA: cek leakage + konsistensi angka laporan/slide vs tabel
```

### Catatan reproduksi

- **Split final** disimpan di `data/processed/splits_{A,B}.csv` (di-commit). `02_preprocess.py` memakai ulang file
  ini bila ada; **jangan** memakai `--resplit`, karena split akan berubah.
- Semua parameter ada di `config.yaml`. `seed = 42` diterapkan ke `random`, NumPy, PyTorch, dan scikit-learn.
  Training CNN di CPU bersifat deterministik pada mesin yang sama, tetapi bisa berbeda sedikit di perangkat lain.
- Eksperimen A: `status` COVID-19 vs healthy, `cough_detected ≥ 0,8`, train di-*undersample* 3 : 1 (hanya train).
  Eksperimen B: *majority vote* `diagnosis_1..4` (COVID-19 vs healthy_cough).
- `scripts/08_qa_final.py` memeriksa bahwa uuid train/val/test saling lepas, normalisasi CNN di-fit hanya pada
  train, scaler/SMOTE berada di dalam *pipeline*, dan angka laporan/slide sama dengan `reports/tables/`.

## Demo inferensi

```powershell
python scripts/predict.py "rekaman batuk.webm"            # model SVM Eksperimen A (default)
python scripts/predict.py batuk.mp3 --eksperimen B         # model berlabel ahli
python scripts/predict.py batuk.wav --semua                # semua model ML
python scripts/predict.py batuk.ogg --json                 # keluaran JSON
```

Script menjalankan preprocessing dan ekstraksi fitur yang **sama** dengan training (hasilnya identik dengan skor
test tersimpan, selisih < 10⁻⁴), lalu menampilkan probabilitas P(COVID-19) dan label berdasarkan ambang dari data
validasi. Format apa pun yang dapat didekode ffmpeg (wav, webm, ogg, mp3, m4a, …) dapat dipakai. Probabilitasnya
tidak terkalibrasi, jadi bandingkan dengan ambang, bukan dengan 0,5. **Bukan alat diagnosis medis.**

## Keterbatasan

Label *self-reported* tanpa PCR; tidak ada ID subjek (split per rekaman); variasi mikrofon dan lingkungan
(*crowdsourced*); bias demografi; *undersampling* train mengubah kalibrasi; test B kecil (152). Rincian ada di
`reports/ringkasan_hasil.md` dan Bab IV–V laporan.

## Lisensi dataset & sitasi

Dataset COUGHVID dirilis dengan lisensi **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Data mentah
tidak disertakan di repositori atau zip pengumpulan; unduh dari Zenodo (record 7024894). Bila memakai dataset ini,
sitasi:

> L. Orlandic, T. Teijeiro, and D. Atienza, "The COUGHVID crowdsourcing dataset, a corpus for the study of
> large-scale cough analysis algorithms," *Scientific Data*, vol. 8, p. 156, 2021, doi: 10.1038/s41597-021-00937-4.

```bibtex
@article{orlandic2021coughvid,
  author  = {Orlandic, Lara and Teijeiro, Tomas and Atienza, David},
  title   = {The {COUGHVID} crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms},
  journal = {Scientific Data},
  volume  = {8},
  pages   = {156},
  year    = {2021},
  doi     = {10.1038/s41597-021-00937-4}
}
```

Referensi utama lain (R1–R4) ada di `references/references.bib` dan Daftar Pustaka laporan.

## Anggota kelompok

| Nama | NIM |
|---|---|
| Meidhika Nawa Sapta | 6250211011 |
| Ari Risnanto | 6250211006 |
| Roslilie Simbulan | 6250211001 |
