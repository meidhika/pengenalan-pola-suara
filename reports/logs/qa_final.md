# QA akhir — Tahap 8

Hasil: **158/158 lulus**.

Dibuat oleh `scripts/08_qa_final.py`. Angka acuan dibaca dari `reports/tables/*.csv` dan `data/processed/*.csv`; teks dicari pada `Laporan_COUGHVID_Kelompok.docx` dan `Presentasi_COUGHVID_Kelompok.pptx`.

## Leakage

| | Pemeriksaan | Detail |
|---|---|---|
| ✅ | A: uuid splits train/val/test saling lepas | irisan=0, uuid unik=True |
| ✅ | A: uuid manifest train/val/test saling lepas | irisan=0, uuid unik=True |
| ✅ | A: val tidak di-undersample (manifest ⊇ populasi split, kecuali gagal preprocessing) | populasi 1628, manifest 1622, hilang 6 (file gagal preprocessing) |
| ✅ | A: prevalensi val alami | populasi 0.0663 vs manifest 0.0666 |
| ✅ | A: test tidak di-undersample (manifest ⊇ populasi split, kecuali gagal preprocessing) | populasi 1628, manifest 1624, hilang 4 (file gagal preprocessing) |
| ✅ | A: prevalensi test alami | populasi 0.0663 vs manifest 0.0665 |
| ✅ | A: fitur tabular = manifest (uuid, split, label) | 5255 baris; tidak cocok=0 |
| ✅ | A: array 2D train = manifest train | n=2009 |
| ✅ | A: array 2D val = manifest val | n=1622 |
| ✅ | A: array 2D test = manifest test | n=1624 |
| ✅ | B: uuid splits train/val/test saling lepas | irisan=0, uuid unik=True |
| ✅ | B: uuid manifest train/val/test saling lepas | irisan=0, uuid unik=True |
| ✅ | B: val tidak di-undersample (manifest ⊇ populasi split, kecuali gagal preprocessing) | populasi 152, manifest 152, hilang 0 (file gagal preprocessing) |
| ✅ | B: prevalensi val alami | populasi 0.4671 vs manifest 0.4671 |
| ✅ | B: test tidak di-undersample (manifest ⊇ populasi split, kecuali gagal preprocessing) | populasi 152, manifest 152, hilang 0 (file gagal preprocessing) |
| ✅ | B: prevalensi test alami | populasi 0.4605 vs manifest 0.4605 |
| ✅ | B: fitur tabular = manifest (uuid, split, label) | 1011 baris; tidak cocok=0 |
| ✅ | B: array 2D train = manifest train | n=707 |
| ✅ | B: array 2D val = manifest val | n=152 |
| ✅ | B: array 2D test = manifest test | n=152 |
| ✅ | ml_dummy_A: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → clf |
| ✅ | ml_dummy_A: skor val tersimpan = ukuran val (ambang dipilih di val) | 1622 vs 1622 |
| ✅ | ml_dummy_B: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → clf |
| ✅ | ml_dummy_B: skor val tersimpan = ukuran val (ambang dipilih di val) | 152 vs 152 |
| ✅ | ml_knn_A: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → clf |
| ✅ | ml_knn_A: skor val tersimpan = ukuran val (ambang dipilih di val) | 1622 vs 1622 |
| ✅ | ml_knn_B: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → clf |
| ✅ | ml_knn_B: skor val tersimpan = ukuran val (ambang dipilih di val) | 152 vs 152 |
| ✅ | ml_logreg_A: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → clf |
| ✅ | ml_logreg_A: skor val tersimpan = ukuran val (ambang dipilih di val) | 1622 vs 1622 |
| ✅ | ml_logreg_B: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → clf |
| ✅ | ml_logreg_B: skor val tersimpan = ukuran val (ambang dipilih di val) | 152 vs 152 |
| ✅ | ml_rf_A: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → clf |
| ✅ | ml_rf_A: skor val tersimpan = ukuran val (ambang dipilih di val) | 1622 vs 1622 |
| ✅ | ml_rf_B: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → smote → clf |
| ✅ | ml_rf_B: skor val tersimpan = ukuran val (ambang dipilih di val) | 152 vs 152 |
| ✅ | ml_svm_A: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → clf |
| ✅ | ml_svm_A: skor val tersimpan = ukuran val (ambang dipilih di val) | 1622 vs 1622 |
| ✅ | ml_svm_B: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → smote → clf |
| ✅ | ml_svm_B: skor val tersimpan = ukuran val (ambang dipilih di val) | 152 vs 152 |
| ✅ | ml_xgb_A: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → clf |
| ✅ | ml_xgb_A: skor val tersimpan = ukuran val (ambang dipilih di val) | 1622 vs 1622 |
| ✅ | ml_xgb_B: scaler/SelectKBest/[SMOTE] di dalam pipeline | scaler → select → clf |
| ✅ | ml_xgb_B: skor val tersimpan = ukuran val (ambang dipilih di val) | 152 vs 152 |
| ✅ | cnn_logmel_A.pt: mean/std normalisasi = statistik train saja (n=2009) | maks /selisih/ vs train 0.00e+00; vs train+val 6.04e-02 |
| ✅ | cnn_mfcc2d_A.pt: mean/std normalisasi = statistik train saja (n=2009) | maks /selisih/ vs train 0.00e+00; vs train+val 1.19e-01 |
| ✅ | Uji silang S1/S2: set evaluasi tanpa uuid train/val model sumber | S1 102 (tabel 102), S2 1554 (tabel 1554) |

## Konsistensi

| | Pemeriksaan | Detail |
|---|---|---|
| ✅ | A Dummy (stratified) ROC-AUC | `0,516` — docx: ya, slide: ya |
| ✅ | A Dummy (stratified) CI 95% | `0,472–0,563` — docx: ya, slide: ya |
| ✅ | A Dummy (stratified) PR-AUC | `0,069` — docx: ya, slide: ya |
| ✅ | A Dummy (stratified) sensitivitas | `1,000` — docx: ya, slide: ya |
| ✅ | A Dummy (stratified) spesifisitas | `0,000` — docx: ya, slide: ya |
| ✅ | A Logistic Regression ROC-AUC | `0,569` — docx: ya, slide: ya |
| ✅ | A Logistic Regression CI 95% | `0,514–0,623` — docx: ya, slide: ya |
| ✅ | A Logistic Regression PR-AUC | `0,087` — docx: ya, slide: ya |
| ✅ | A Logistic Regression sensitivitas | `0,481` — docx: ya, slide: ya |
| ✅ | A Logistic Regression spesifisitas | `0,600` — docx: ya, slide: ya |
| ✅ | A SVM (RBF) ROC-AUC | `0,576` — docx: ya, slide: ya |
| ✅ | A SVM (RBF) CI 95% | `0,523–0,629` — docx: ya, slide: ya |
| ✅ | A SVM (RBF) PR-AUC | `0,088` — docx: ya, slide: ya |
| ✅ | A SVM (RBF) sensitivitas | `0,556` — docx: ya, slide: ya |
| ✅ | A SVM (RBF) spesifisitas | `0,553` — docx: ya, slide: ya |
| ✅ | A k-NN ROC-AUC | `0,580` — docx: ya, slide: ya |
| ✅ | A k-NN CI 95% | `0,525–0,636` — docx: ya, slide: ya |
| ✅ | A k-NN PR-AUC | `0,088` — docx: ya, slide: ya |
| ✅ | A k-NN sensitivitas | `0,509` — docx: ya, slide: ya |
| ✅ | A k-NN spesifisitas | `0,613` — docx: ya, slide: ya |
| ✅ | A Random Forest ROC-AUC | `0,570` — docx: ya, slide: ya |
| ✅ | A Random Forest CI 95% | `0,513–0,628` — docx: ya, slide: ya |
| ✅ | A Random Forest PR-AUC | `0,091` — docx: ya, slide: ya |
| ✅ | A Random Forest sensitivitas | `0,407` — docx: ya, slide: ya |
| ✅ | A Random Forest spesifisitas | `0,705` — docx: ya, slide: ya |
| ✅ | A XGBoost ROC-AUC | `0,572` — docx: ya, slide: ya |
| ✅ | A XGBoost CI 95% | `0,513–0,629` — docx: ya, slide: ya |
| ✅ | A XGBoost PR-AUC | `0,097` — docx: ya, slide: ya |
| ✅ | A XGBoost sensitivitas | `0,630` — docx: ya, slide: ya |
| ✅ | A XGBoost spesifisitas | `0,433` — docx: ya, slide: ya |
| ✅ | A CNN (log-mel) ROC-AUC | `0,567` — docx: ya, slide: ya |
| ✅ | A CNN (log-mel) CI 95% | `0,511–0,619` — docx: ya, slide: ya |
| ✅ | A CNN (log-mel) PR-AUC | `0,081` — docx: ya, slide: ya |
| ✅ | A CNN (log-mel) sensitivitas | `0,370` — docx: ya, slide: ya |
| ✅ | A CNN (log-mel) spesifisitas | `0,708` — docx: ya, slide: ya |
| ✅ | A CNN (MFCC 2D) ROC-AUC | `0,544` — docx: ya, slide: ya |
| ✅ | A CNN (MFCC 2D) CI 95% | `0,487–0,601` — docx: ya, slide: ya |
| ✅ | A CNN (MFCC 2D) PR-AUC | `0,086` — docx: ya, slide: ya |
| ✅ | A CNN (MFCC 2D) sensitivitas | `0,463` — docx: ya, slide: ya |
| ✅ | A CNN (MFCC 2D) spesifisitas | `0,596` — docx: ya, slide: ya |
| ✅ | B Dummy (stratified) ROC-AUC | `0,487` — docx: ya, slide: ya (opsional) |
| ✅ | B Dummy (stratified) CI 95% | `0,417–0,567` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B Dummy (stratified) PR-AUC | `0,454` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B Dummy (stratified) sensitivitas | `1,000` — docx: ya, slide: ya (opsional) |
| ✅ | B Dummy (stratified) spesifisitas | `0,000` — docx: ya, slide: ya (opsional) |
| ✅ | B Logistic Regression ROC-AUC | `0,812` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B Logistic Regression CI 95% | `0,730–0,877` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B Logistic Regression PR-AUC | `0,749` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B Logistic Regression sensitivitas | `0,771` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B Logistic Regression spesifisitas | `0,817` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B SVM (RBF) ROC-AUC | `0,826` — docx: ya, slide: ya |
| ✅ | B SVM (RBF) CI 95% | `0,745–0,888` — docx: ya, slide: ya |
| ✅ | B SVM (RBF) PR-AUC | `0,786` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B SVM (RBF) sensitivitas | `0,743` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B SVM (RBF) spesifisitas | `0,805` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B k-NN ROC-AUC | `0,776` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B k-NN CI 95% | `0,689–0,848` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B k-NN PR-AUC | `0,675` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B k-NN sensitivitas | `0,629` — docx: ya, slide: ya (opsional) |
| ✅ | B k-NN spesifisitas | `0,756` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B Random Forest ROC-AUC | `0,817` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B Random Forest CI 95% | `0,741–0,879` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B Random Forest PR-AUC | `0,771` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B Random Forest sensitivitas | `0,800` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B Random Forest spesifisitas | `0,732` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B XGBoost ROC-AUC | `0,807` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B XGBoost CI 95% | `0,726–0,871` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B XGBoost PR-AUC | `0,754` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B XGBoost sensitivitas | `0,614` — docx: ya, slide: TIDAK (opsional) |
| ✅ | B XGBoost spesifisitas | `0,817` — docx: ya, slide: TIDAK (opsional) |
| ✅ | A n test | `1.624` — docx: ya, slide: ya |
| ✅ | A prevalensi test | `0,067` — docx: ya, slide: ya |
| ✅ | CNN log-mel mean±std 3 seed | `0,572 ± 0,006` — docx: ya, slide: ya |
| ✅ | CNN MFCC mean±std 3 seed | `0,579 ± 0,030` — docx: ya, slide: ya |
| ✅ | p DeLong SVM vs dummy (A) | `0,12` — docx: ya, slide: ya |
| ✅ | p DeLong A k-NN vs SVM (RBF) | `0,87` — docx: ya, slide: TIDAK (opsional) |
| ✅ | p DeLong A SVM (RBF) vs CNN (log-mel) | `0,70` — docx: ya, slide: ya (opsional) |
| ✅ | p DeLong A SVM (RBF) vs CNN (MFCC 2D) | `0,39` — docx: ya, slide: ya (opsional) |
| ✅ | p DeLong A SVM (RBF) vs Dummy (stratified) | `0,12` — docx: ya, slide: ya (opsional) |
| ✅ | p DeLong B SVM (RBF) vs Random Forest | `0,62` — docx: ya, slide: ya (opsional) |
| ✅ | Kesepakatan label n | `572` — docx: ya, slide: ya |
| ✅ | Kesepakatan label % | `50,3` — docx: ya, slide: ya |
| ✅ | % self-reported COVID dinilai COVID oleh ahli | `43%` — docx: ya (opsional), slide: ya |
| ✅ | Uji silang SVM S1 ← label self-reported (A) | `0,673` — docx: ya, slide: TIDAK (opsional) |
| ✅ | Uji silang SVM S1 ← label self-reported (A) (2 desimal, slide) | `0,67` — docx: ya (opsional), slide: ya |
| ✅ | Uji silang SVM S1 ← label ahli (B) | `0,896` — docx: ya, slide: TIDAK (opsional) |
| ✅ | Uji silang SVM S1 ← label ahli (B) (2 desimal, slide) | `0,90` — docx: ya (opsional), slide: ya |
| ✅ | Uji silang SVM S2 ← label self-reported (A) | `0,585` — docx: ya, slide: TIDAK (opsional) |
| ✅ | Uji silang SVM S2 ← label ahli (B) | `0,545` — docx: ya, slide: TIDAK (opsional) |
| ✅ | Uji silang SVM S3 ← label self-reported (A) | `0,574` — docx: ya, slide: TIDAK (opsional) |
| ✅ | Uji silang SVM S3 ← label self-reported (A) | `0,541` — docx: ya, slide: TIDAK (opsional) |
| ✅ | Median skor SVM healthy | `0,242` — docx: ya, slide: ya |
| ✅ | Median skor SVM COVID-19 | `0,254` — docx: ya, slide: ya |
| ✅ | Spearman skor–durasi SVM (RBF) | `0,41` — docx: ya, slide: ya |
| ✅ | Spearman skor–durasi CNN (log-mel) | `0,61` — docx: ya, slide: ya |
| ✅ | Durasi median FP | `5,0` — docx: ya, slide: ya |
| ✅ | Durasi median TN | `3,0` — docx: ya, slide: ya |
| ✅ | AUC subgrup demam ya | `0,51` — docx: ya (opsional), slide: ya |
| ✅ | AUC subgrup demam ya (3 desimal) | `0,507` — docx: ya, slide: TIDAK (opsional) |
| ✅ | AUC subgrup demam tidak | `0,59` — docx: ya (opsional), slide: ya |
| ✅ | AUC subgrup demam tidak (3 desimal) | `0,590` — docx: ya, slide: TIDAK (opsional) |
| ✅ | AUC subgrup umur ≥ 55 | `0,47` — docx: ya (opsional), slide: ya |
| ✅ | Jumlah fitur signifikan A | `39 / 315` — docx: TIDAK (opsional), slide: ya |
| ✅ | Jumlah fitur | `315` — docx: ya, slide: ya |
| ✅ | AUC fitur terkuat (contrast_05_std) | `0,584` — docx: ya, slide: ya |
| ✅ | Parameter LightCNN | `97.521` — docx: ya, slide: ya |
| ✅ | Total rekaman | `34.434` — docx: ya, slide: ya |
| ✅ | Populasi A COVID-19 | `720` — docx: ya, slide: ya |
| ✅ | Populasi A healthy | `10.132` — docx: ya, slide: ya |
| ✅ | Manifest B COVID-19 | `470` — docx: ya, slide: ya |
| ✅ | Manifest B healthy | `541` — docx: ya, slide: ya |
