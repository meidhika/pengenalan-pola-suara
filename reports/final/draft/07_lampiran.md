# LAMPIRAN

## Lampiran A. Struktur Repositori

```
PPKB/
├── CLAUDE.md, PROGRESS.md, README.md   ← konteks proyek & log tiap tahap
├── requirements.txt                   ← daftar pustaka Python
├── config.yaml                        ← semua parameter (sr, durasi, n_mfcc, seed, path, ambang, dst.)
├── The COUGHVID crowdsourcing dataset.zip
├── data/
│   ├── raw/coughvid_20211012/         ← hasil ekstrak zip (read-only)
│   ├── interim/                       ← WAV 16 kHz mono hasil konversi ffmpeg
│   ├── processed/                     ← WAV ternormalisasi, manifest_{A,B}.csv, splits_{A,B}.csv, segments.csv
│   └── features/                      ← tabular_{A,B}.parquet, logmel_*/mfcc2d_*.npy, labels_*, uuids_*
├── src/                               ← seluruh logika (modul yang dapat di-import)
│   ├── config.py        (loader config + set_seed)
│   ├── data_utils.py    (metadata, label A/B/C, split, undersampling)
│   ├── preprocessing.py (konversi, trim, segmentasi, reduksi derau, normalisasi)
│   ├── features.py      (fitur tabular & 2D, uji Mann–Whitney + FDR)
│   ├── models_ml.py     (pipeline ML klasik, ruang hyperparameter, subset ablasi)
│   ├── models_cnn.py    (dataset/augmentasi, LightCNN, CRNN, Grad-CAM)
│   ├── evaluate.py      (metrik, bootstrap CI, Youden, DeLong, subgrup)
│   ├── analysis.py      (tabel final, uji silang label, subgrup)
│   └── plots.py         (semua fungsi grafik)
├── scripts/                           ← entry point CLI per tahap (01_ … 07_)
├── notebooks/                         ← notebook naratif per tahap (sudah dieksekusi)
├── models/                            ← model tersimpan (.joblib, .pt) + prediksi val/test (.parquet)
├── reports/
│   ├── figures/                       ← gambar PNG (dpi 200)
│   ├── tables/                        ← tabel hasil CSV/Markdown
│   ├── logs/                          ← log proses, ringkasan JSON, pip_freeze.txt
│   └── final/                         ← draf laporan (draft/), dokumen akhir
└── references/                        ← PDF referensi + references.bib
```

## Lampiran B. Cara Menjalankan

Semua perintah dijalankan dari *root* proyek di PowerShell setelah *virtual environment* diaktifkan. Setiap script
mendukung `--limit N` untuk uji cepat dan dapat dilanjutkan (*resume*) bila terputus.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # PyTorch versi CPU: --index-url https://download.pytorch.org/whl/cpu
# ffmpeg harus tersedia di PATH

python scripts/01_extract.py                      # ekstrak zip -> data/raw/
python scripts/02_preprocess.py --experiments A B # split + konversi, trim, segmentasi, derau, normalisasi
python scripts/03_extract_features.py             # fitur tabular (315) + log-mel & MFCC 2D
python scripts/04_train_ml.py --experiments A B   # LR, SVM, k-NN, RF, XGBoost + ablasi + importance
python scripts/05_train_cnn.py --features logmel mfcc2d --seeds 42 7 2024   # LightCNN (CPU)
python scripts/06_evaluate_all.py                 # tabel final, DeLong, uji silang label, error, subgrup
python scripts/07_diagram_alur.py                 # diagram alur penelitian (Gambar 1)
```

Notebook `notebooks/01_…06_*.ipynb` memanggil fungsi di `src/` dan dapat dieksekusi ulang dengan
`jupyter nbconvert --to notebook --execute --inplace <notebook>` (kernel `ppkb`).

## Lampiran C. *Hyperparameter* Lengkap

### C.1 *Hyperparameter* terpilih model ML klasik

Hasil `RandomizedSearchCV` (20 iterasi, ROC-AUC) pada train. "Imbalance" adalah varian yang dipilih berdasarkan AUC
CV, sedangkan "ambang" adalah ambang Youden dari val.

**Tabel L1.** *Hyperparameter* terpilih model ML klasik.

| Eksp. | Model | Imbalance | *Hyperparameter* terpilih | Ambang |
|---|---|---|---|---|
| A | *Logistic Regression* | *balanced* | $C$ = 8,26 × 10⁻⁴; k = semua | 0,501 |
| A | SVM (RBF) | *balanced* | $C$ = 2,80; $\gamma$ = 2,62 × 10⁻⁴; k = semua | 0,249 |
| A | k-NN | – | $k$ = 111; bobot *uniform*; $p$ = 2; k fitur = 20 | 0,270 |
| A | *Random Forest* | *balanced* | `max_depth` 5; `min_samples_leaf` 2; `max_features` 0,1; k = 100 | 0,493 |
| A | XGBoost | *balanced* (`scale_pos_weight`) | `max_depth` 4; *lr* 0,0197; `n_estimators` 500; `subsample` 0,6; `colsample_bytree` 0,5; `min_child_weight` 1; k = 100 | 0,320 |
| B | *Logistic Regression* | *balanced* | $C$ = 4,66 × 10⁻³; k = semua | 0,554 |
| B | SVM (RBF) | SMOTE | $C$ = 2,80; $\gamma$ = 2,62 × 10⁻⁴; k = semua | 0,557 |
| B | k-NN | – | $k$ = 104; bobot *distance*; $p$ = 2; k fitur = 100 | 0,539 |
| B | *Random Forest* | SMOTE | `max_depth` 10; `min_samples_leaf` 1; `max_features` 0,1; k = semua | 0,488 |
| B | XGBoost | *balanced* | `max_depth` 4; *lr* 0,0135; `n_estimators` 300; `subsample` 0,6; `colsample_bytree` 0,8; `min_child_weight` 1; k = semua | 0,586 |

<!-- sumber: reports/tables/04_imbalance_A.csv, 04_imbalance_B.csv (kolom best_params, baris dipilih=True); ambang dari 06_tabel_final_{A,B}.csv -->

### C.2 *Hyperparameter* CNN

**Tabel L2.** *Hyperparameter* pelatihan LightCNN.

| Parameter | Nilai |
|---|---|
| Input | *log-mel* 64 × 500 atau MFCC 2D 40 × 500 (10 s, resolusi 20 ms); normalisasi *mean/std* per pita (train) |
| Arsitektur | 4 blok [Conv 3×3 → BN → ReLU → MaxPool 2×2 → *Dropout*], kanal 16-32-64-128, *dropout* 0,10 → 0,20; GAP; *dropout* 0,30; Linear → 1 |
| Jumlah parameter | 97.521 (varian $1\times12$: 129.825; 5 blok: 245.233; CRNN: 245.489, tidak dilatih) |
| *Loss* | `BCEWithLogitsLoss`, `pos_weight` = 3,0 |
| *Optimizer* | AdamW, *lr* 10⁻³, *weight decay* 10⁻² |
| *Scheduler* | ReduceLROnPlateau (AUC val), faktor 0,5, *patience* 4 |
| *Epoch* / *early stopping* | maks. 100 / AUC val, *patience* 10 |
| *Batch size* | 32 (CPU, 6 *thread*) |
| Augmentasi (train) | *random crop* 256 *frame* di area bersuara; *time shift* ±20 *frame*; SpecAugment 2 *freq mask* (≤ 8 pita *log-mel* / ≤ 5 koef. MFCC) + 2 *time mask* (≤ 25 *frame*); *mixup* tidak dipakai |
| *Seed* | 42 (baris utama), 7, 2024 |
| Ambang | Youden di val (*log-mel* *seed* 42: 0,407; MFCC 2D *seed* 42: 0,362) |

<!-- sumber: PROGRESS.md Tahap 5; scripts/05_train_cnn.py; 06_tabel_final_A.csv -->

### C.3 Rincian lapisan LightCNN (input *log-mel*, *crop* 64 × 256)

**Tabel L3.** Rincian lapisan LightCNN.

| Lapisan | Tipe | Bentuk keluaran | Parameter |
|---|---|---|---|
| Blok 1 | Conv2d 3×3 (1→16) + BatchNorm2d + ReLU + MaxPool2d + Dropout | 16 × 32 × 128 | 144 + 32 |
| Blok 2 | Conv2d 3×3 (16→32) + BatchNorm2d + ReLU + MaxPool2d + Dropout | 32 × 16 × 64 | 4.608 + 64 |
| Blok 3 | Conv2d 3×3 (32→64) + BatchNorm2d + ReLU + MaxPool2d + Dropout | 64 × 8 × 32 | 18.432 + 128 |
| Blok 4 | Conv2d 3×3 (64→128) + BatchNorm2d + ReLU + MaxPool2d + Dropout | 128 × 4 × 16 | 73.728 + 256 |
| *Head* | AdaptiveAvgPool2d → Flatten → Dropout → Linear (128→1) | 1 | 129 |
| **Total** | | | **97.521** |

<!-- sumber: reports/tables/05_arsitektur_lightcnn.csv, 05_jumlah_parameter.csv -->

## Lampiran D. Gambar Tambahan

Gambar berikut tersedia di `reports/figures/` dan dapat dilampirkan sesuai kebutuhan:
`01_hist_cough_detected.png` (histogram skor `cough_detected`), `01_label_ahli.png` (distribusi diagnosis
pulmonolog), `01_missing_per_kolom.png`, `02_langkah_preprocessing_healthy.png`,
`02_spektrogram_sebelum_sesudah_noise.png`, `02_distribusi_durasi.png`, `03_violin_fitur_kunci.png`,
`03_distribusi_auc_fitur.png`, `04_confusion_A.png`, `04_confusion_B.png`, `04_importance_B.png`, `04_ablasi_B.png`,
`05_augmentasi.png`, `05_kurva_training_mfcc2d.png`, `05_perbandingan_auc.png`, `05_roc_pr_A.png`,
`05_confusion_A.png`, `05_gradcam_profil_frekuensi.png`, dan `06_subgrup_cnn_logmel_A.png`.

## Lampiran E. Pembagian Tugas Anggota Kelompok

**Tabel L4.** Pembagian tugas anggota kelompok.

| No. | Nama | NIM | Tugas |
|---|---|---|---|
| 1 | Meidhika Nawa Sapta | 6250211011 | Akuisisi data, *preprocessing*, ekstraksi fitur (Tahap 1–3) |
| 2 | Ari Risnanto | 6250211006 | Klasifikasi ML klasik & CNN (Tahap 4–5) |
| 3 | Roslilie Simbulan | 6250211001 | Evaluasi, analisis, penyusunan laporan & presentasi (Tahap 6–7) |

*Catatan: pembagian tugas di atas adalah usulan dan dapat disesuaikan oleh kelompok.*
