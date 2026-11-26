# Daftar Figur & Tabel Final

Penomoran mengikuti urutan pipeline (slide 10): akuisisi → preprocessing → fitur → klasifikasi → evaluasi.
Figur: `reports/figures/` (PNG, dpi 200). Tabel: `reports/tables/` (CSV; versi Markdown `.md` bila ada).
Kolom **Inti** (✓) = disarankan masuk badan laporan; sisanya untuk lampiran.

## Gambar

| No. | File | Inti | Caption |
|---|---|---|---|
| Gambar 1 | `01_contoh_waveform.png` | ✓ | Contoh *waveform* (amplitudo terhadap waktu) rekaman batuk asli untuk tiap status *self-reported*, beserta codec, *sampling rate* asli, dan durasinya. |
| Gambar 2 | `01_distribusi_label_demografi.png` | ✓ | Distribusi `status` (*self-reported*), `status_SSL`, gender, kondisi pernapasan, demam/nyeri otot, dan usia pada metadata COUGHVID v3. |
| Gambar 3 | `01_hist_cough_detected.png` |  | Histogram skor `cough_detected`; garis putus-putus = ambang 0,8 yang direkomendasikan penyusun dataset. |
| Gambar 4 | `01_label_ahli.png` |  | Distribusi diagnosis pulmonolog (`diagnosis_1..4`) pada subset berlabel ahli. |
| Gambar 5 | `01_missing_per_kolom.png` |  | Proporsi nilai kosong per kolom metadata. |
| Gambar 6 | `02_langkah_preprocessing_covid.png` | ✓ | Tahapan preprocessing pada contoh rekaman COVID-19: konversi 16 kHz mono, *trimming*, segmentasi batuk, reduksi *noise*, normalisasi. |
| Gambar 7 | `02_langkah_preprocessing_healthy.png` |  | Tahapan preprocessing pada contoh rekaman healthy. |
| Gambar 8 | `02_spektrogram_sebelum_sesudah_noise.png` |  | Spektrogram sebelum dan sesudah reduksi *noise* (*spectral gating*). |
| Gambar 9 | `02_distribusi_durasi.png` |  | Distribusi durasi rekaman asli, setelah *trim*, dan total segmen batuk. |
| Gambar 10 | `02_boxplot_durasi.png` |  | Perbandingan durasi setelah *trim* per kelas pada Eksperimen A dan B. |
| Gambar 11 | `02_jumlah_per_split.png` | ✓ | Jumlah rekaman per split (train/val/test) dan per kelas untuk Eksperimen A (train di-*undersample* 3:1) dan B. |
| Gambar 12 | `03_waveform_covid_vs_healthy.png` | ✓ | *Waveform* batuk COVID-19 vs healthy setelah preprocessing (domain waktu; batuk bersifat nonperiodik). |
| Gambar 13 | `03_spektrum_fft_frame.png` | ✓ | Spektrum magnitudo FFT satu *frame* 32 ms (domain frekuensi) untuk batuk COVID-19 dan healthy. |
| Gambar 14 | `03_representasi_waktu_frekuensi.png` | ✓ | Representasi waktu–frekuensi satu rekaman: spektrogram STFT, log-mel, MFCC, dan GFCC. |
| Gambar 15 | `03_rata2_logmel_per_kelas.png` |  | Rata-rata log-mel spectrogram per kelas dan selisihnya (COVID-19 − healthy). |
| Gambar 16 | `03_violin_fitur_kunci.png` |  | Distribusi fitur tabular paling diskriminatif per kelas (Eksperimen A). |
| Gambar 17 | `03_distribusi_auc_fitur.png` |  | Distribusi AUC univariat 315 fitur tabular (Eksperimen A). |
| Gambar 18 | `03_proyeksi_pca_tsne.png` |  | Proyeksi PCA dan t-SNE fitur tabular; kedua kelas sangat tumpang-tindih. |
| Gambar 19 | `04_roc_pr_A.png` | ✓ | Kurva ROC dan Precision–Recall model ML klasik, Eksperimen A (test). |
| Gambar 20 | `04_roc_pr_B.png` | ✓ | Kurva ROC dan Precision–Recall model ML klasik, Eksperimen B (test, label ahli). |
| Gambar 21 | `04_confusion_A.png` |  | *Confusion matrix* SVM (RBF) Eksperimen A pada ambang Youden (val) dan ambang 0,5. |
| Gambar 22 | `04_confusion_B.png` |  | *Confusion matrix* SVM (RBF) Eksperimen B pada ambang Youden (val) dan ambang 0,5. |
| Gambar 23 | `04_importance_A.png` |  | Fitur terpenting Eksperimen A (RF, XGBoost, *permutation importance* SVM). |
| Gambar 24 | `04_importance_B.png` |  | Fitur terpenting Eksperimen B. |
| Gambar 25 | `04_ablasi_A.png` | ✓ | Ablasi kelompok fitur (waktu, spektral, MFCC, GFCC, gabungan) pada SVM, Eksperimen A. |
| Gambar 26 | `04_ablasi_B.png` |  | Ablasi kelompok fitur pada SVM, Eksperimen B. |
| Gambar 27 | `05_arsitektur.png` | ✓ | Arsitektur LightCNN (4 blok konvolusi 3×3, *global average pooling*, 97.521 parameter). |
| Gambar 28 | `05_augmentasi.png` |  | Contoh augmentasi train: *random crop*, *time shift*, dan SpecAugment pada log-mel. |
| Gambar 29 | `05_kurva_training_logmel.png` | ✓ | Kurva *loss* dan AUC train/val CNN log-mel (3 seed). |
| Gambar 30 | `05_kurva_training_mfcc2d.png` |  | Kurva *loss* dan AUC train/val CNN MFCC 2D (3 seed). |
| Gambar 31 | `05_perbandingan_auc.png` |  | ROC-AUC val dan test CNN log-mel vs MFCC 2D per seed (mean ± std) dibanding SVM (RBF). |
| Gambar 32 | `05_roc_pr_A.png` |  | Kurva ROC/PR CNN log-mel dan MFCC 2D, Eksperimen A. |
| Gambar 33 | `05_confusion_A.png` |  | *Confusion matrix* CNN (seed 42), Eksperimen A. |
| Gambar 34 | `05_gradcam.png` | ✓ | Peta Grad-CAM CNN log-mel (seed 42) pada rekaman test: per kelas, satu contoh benar paling yakin dan satu contoh acak. |
| Gambar 35 | `05_gradcam_profil_frekuensi.png` |  | Profil frekuensi rata-rata Grad-CAM per kelas. |
| Gambar 36 | `06_roc_gabungan_A.png` | ✓ | Kurva ROC dan Precision–Recall gabungan Eksperimen A (test): ML terbaik (SVM RBF), CNN log-mel, CNN MFCC 2D, dan baseline *dummy*. |
| Gambar 37 | `06_uji_silang_label.png` | ✓ | Uji silang sumber label (SVM RBF): AUC model berlabel *self-reported* (A) vs berlabel ahli (B) pada set evaluasi berlabel ahli (S1), *self-reported* (S2), dan rekaman yang sama dengan dua label (S3). |
| Gambar 38 | `06_distribusi_skor_A.png` | ✓ | Distribusi skor prediksi P(COVID-19) per kelas pada test A untuk SVM, CNN log-mel, dan CNN MFCC 2D; garis putus-putus = ambang Youden dari val. |
| Gambar 39 | `06_contoh_fp_fn_A.png` | ✓ | Log-mel spectrogram tiga *false positive* (atas) dan tiga *false negative* (bawah) paling yakin dari SVM (RBF), Eksperimen A. |
| Gambar 40 | `06_subgrup_svm_A.png` | ✓ | ROC-AUC [CI 95%] SVM (RBF) per subgrup demografi dan gejala, Eksperimen A (test). |
| Gambar 41 | `06_subgrup_cnn_logmel_A.png` |  | ROC-AUC [CI 95%] CNN log-mel per subgrup, Eksperimen A (test). |

## Tabel

| No. | File | Inti | Caption |
|---|---|---|---|
| Tabel 1 | `01_ringkasan_angka.csv` | ✓ | Ringkasan dataset COUGHVID v3 (jumlah rekaman, format audio, label tersedia). |
| Tabel 2 | `01_crosstab_status_x_diagnosis_ahli.csv` |  | Tabulasi silang status *self-reported* vs diagnosis ahli pada metadata. |
| Tabel 3 | `02_jumlah_per_split_A.csv` / `02_jumlah_per_split_B.csv` | ✓ | Jumlah rekaman per split dan kelas untuk Eksperimen A dan B. |
| Tabel 4 | `02_ringkasan_durasi.csv` |  | Median durasi rekaman (asli, setelah *trim*, segmen batuk) per kelas. |
| Tabel 5 | `03_uji_fitur.csv` |  | 15 fitur tabular teratas menurut uji Mann-Whitney (FDR) dan AUC univariat, Eksperimen A. |
| Tabel 6 | `04_ablasi_fitur.csv` / `04_ablasi_fitur_B.csv` |  | Hasil ablasi kelompok fitur (AUC CV, val, test) untuk SVM. |
| Tabel 7 | `05_arsitektur_lightcnn.csv` |  | Rincian layer LightCNN: tipe, bentuk keluaran, jumlah parameter. |
| Tabel 8 | `05_cnn_per_seed.csv` |  | Metrik val dan test CNN per fitur dan seed. |
| Tabel 9 | `06_tabel_final_A.csv` / `.md` | ✓ | Performa final semua model Eksperimen A pada test (n = 1.624; prevalensi 0,067): Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, ROC-AUC [CI 95%], PR-AUC. ★ = AUC val tertinggi, ▲ = AUC test tertinggi. |
| Tabel 10 | `06_tabel_final_B.csv` / `.md` | ✓ | Performa final model ML klasik Eksperimen B (label ahli) pada test (n = 152). |
| Tabel 11 | `06_uji_signifikansi_auc.csv` / `.md` | ✓ | Uji signifikansi selisih ROC-AUC (DeLong dan bootstrap berpasangan) antar-model pada test yang sama. |
| Tabel 12 | `06_kesepakatan_label_B.csv` / `.md` | ✓ | Tabulasi silang status *self-reported* vs diagnosis ahli pada rekaman Eksperimen B (Cohen's κ = 0,00). |
| Tabel 13 | `06_uji_silang_label.csv` / `.md` | ✓ | ROC-AUC [CI 95%] uji silang: model yang dilatih dengan label *self-reported* (A) vs label ahli (B), dievaluasi pada set S1–S3 tanpa uuid train/val model sumber. |
| Tabel 14 | `06_karakteristik_error_A.csv` / `.md` |  | Karakteristik rekaman per jenis keputusan SVM (TP, FN, FP, TN): durasi, `cough_detected`, gejala, gender. |
| Tabel 15 | `06_korelasi_skor_durasi_A.csv` |  | Korelasi Spearman antara skor model dan durasi rekaman setelah *trim*. |
| Tabel 16 | `06_subgrup_fairness.csv` / `.md` | ✓ | Performa per subgrup (gender, kelompok umur, kondisi pernapasan, demam/nyeri otot) pada test A: n, positif, ROC-AUC [CI 95%], sensitivitas, spesifisitas. |
| Tabel 17 | `06_distribusi_skor_A.csv` |  | Statistik distribusi skor prediksi per kelas dan proporsi di atas ambang. |
| Tabel 18 | `06_perbandingan_literatur.csv` / `.md` | ✓ | Perbandingan dengan literatur (R0, R2, R3, R4): dataset, label, fitur, model, protokol, dan metrik yang dilaporkan vs penelitian ini. |
