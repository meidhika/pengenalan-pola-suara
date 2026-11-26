# TAHAP 5 — Klasifikasi dengan CNN Ringan (Deep Learning)

> Ketik `/clear` dulu.

---

```
Baca CLAUDE.md dan PROGRESS.md. Kita mengerjakan TAHAP 5 saja: CNN ringan pada fitur 2D
(log-mel spectrogram dan MFCC 2D) untuk Eksperimen A. Gunakan src/evaluate.py yang sudah ada
agar metrik sebanding dengan model ML klasik.

Langkah:
1. Cek device (CUDA/CPU) dan sesuaikan batch size. Pastikan training bisa jalan di CPU dalam
   waktu wajar (opsi --epochs, --limit untuk uji cepat).
2. `src/models_cnn.py` (PyTorch):
   - Dataset class yang membaca .npy (memmap bila besar) + augmentasi HANYA pada train:
     SpecAugment (time & frequency masking), random time shift, mixup opsional; augmentasi
     di level waveform (noise, pitch shift, time stretch seperti R2) boleh ditambahkan sebagai
     opsi offline jika waktu memungkinkan.
   - Model `LightCNN` terinspirasi R3: 4–5 blok [Conv2d → BatchNorm → ReLU → MaxPool → Dropout],
     filter 16-32-64-128, Global Average Pooling, FC → 1 output (sigmoid / BCEWithLogits).
     Sediakan variasi kernel (3×3 vs 1×12 ala R3) sebagai parameter. Cetak jumlah parameter.
   - (Opsional, bila waktu cukup) model kedua: CRNN kecil (CNN + GRU) sebagai pembanding.
3. `scripts/05_train_cnn.py`:
   - Loss BCEWithLogits dengan pos_weight (imbalance) atau WeightedRandomSampler — pilih satu,
     dokumentasikan alasannya.
   - Optimizer AdamW, lr 1e-3, scheduler ReduceLROnPlateau/OneCycle, maks 100 epoch,
     early stopping pada val AUC (patience 10), simpan checkpoint terbaik `models/cnn_<fitur>_A.pt`.
   - Log per epoch (loss, val AUC) ke CSV `reports/logs/cnn_<fitur>_history.csv`.
   - Threshold dipilih di VAL (Youden), evaluasi sekali di TEST, simpan via save_results().
   - Ulangi dengan 3 seed (42, 7, 2024) → laporkan mean ± std AUC.
   Jalankan untuk input log-mel dan MFCC 2D. Uji --limit dan 2 epoch dulu.
4. Notebook `notebooks/05_cnn.ipynb`: diagram/ringkasan arsitektur (tabel layer & parameter),
   kurva loss dan AUC train/val per epoch, ROC & confusion matrix test, perbandingan log-mel vs
   MFCC, dan (opsional) Grad-CAM pada 2 contoh per kelas untuk melihat area spectrogram yang
   diperhatikan model. Figur → `reports/figures/05_*.png`. Eksekusi notebook.
5. Perbarui PROGRESS.md: hasil CNN (mean ± std), lama training, apakah terjadi overfitting,
   perbandingan singkat dengan model ML terbaik. Berhenti dan ringkas.
```
