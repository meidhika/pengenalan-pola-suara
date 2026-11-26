# TAHAP 4 — Klasifikasi dengan Machine Learning Klasik

> Ketik `/clear` dulu.

---

```
Baca CLAUDE.md dan PROGRESS.md. Kita mengerjakan TAHAP 4 saja: model ML klasik pada fitur tabular
Eksperimen A (lalu B).

Langkah:
1. Buat `src/evaluate.py` berisi fungsi metrik yang dipakai SEMUA model (termasuk CNN nanti):
   accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC,
   confusion matrix, kurva ROC & PR, serta 95% CI AUC dengan bootstrap (1000x) di test set.
   Fungsi `save_results(model_name, experiment, metrics)` menambahkan baris ke
   `reports/tables/hasil_semua_model.csv`.
2. Buat `src/models_ml.py` dengan pipeline sklearn/imblearn (Scaler → [SelectKBest opsional] →
   classifier), model:
   - Baseline: DummyClassifier (stratified) — penting sebagai pembanding.
   - Logistic Regression, SVM (RBF, probability=True), k-NN, Random Forest, XGBoost.
   Penanganan imbalance (bandingkan): class_weight='balanced' vs SMOTE di dalam pipeline imblearn
   (hanya mengenai train fold).
3. Buat `scripts/04_train_ml.py`:
   - Tuning hyperparameter dengan StratifiedKFold 5-fold pada TRAIN (+VAL digabung untuk CV,
     atau train saja lalu val untuk pemilihan — pilih satu & dokumentasikan), metrik scoring ROC-AUC.
     Grid kecil/RandomizedSearchCV (n_iter ≤ 20) agar cepat.
   - Pilih threshold keputusan pada VAL (maksimalkan Youden's J), lalu evaluasi SEKALI di TEST.
   - Simpan model terbaik tiap algoritma ke `models/ml_<nama>_A.joblib`.
   - Ablasi fitur dengan model terbaik: (i) hanya waktu (RMS, ZCR), (ii) hanya spektral,
     (iii) hanya MFCC, (iv) hanya GFCC, (v) MFCC+GFCC, (vi) semua → `reports/tables/04_ablasi_fitur.csv`.
4. Jalankan untuk Eksperimen A, lalu Eksperimen B (data kecil → pakai repeated stratified CV,
   laporkan mean ± std).
5. Notebook `notebooks/04_ml_klasik.ipynb`: tabel perbandingan model, kurva ROC semua model dalam
   satu plot, confusion matrix model terbaik, feature importance (RF/XGBoost, permutation
   importance) top-20, hasil ablasi (bar chart). Figur → `reports/figures/04_*.png`. Eksekusi.
6. Perbarui PROGRESS.md: tabel ringkas hasil test (AUC, F1, sensitivity, specificity) per model,
   model terbaik, fitur terpenting, catatan interpretasi. Berhenti dan ringkas.
```
