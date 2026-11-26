# TAHAP 2 — Preprocessing Audio dan Penetapan Dataset Eksperimen

> Sebelum menempel: ketik `/clear` di Claude Code untuk mengosongkan context.

---

```
Baca CLAUDE.md dan PROGRESS.md. Kita mengerjakan TAHAP 2 saja: preprocessing audio dan pembuatan
manifest dataset. Jangan ekstraksi fitur dulu.

Langkah:
1. Tetapkan populasi data di `src/data_utils.py` → fungsi `build_label_table()`:
   - Eksperimen A: cough_detected ≥ threshold, status ∈ {COVID-19, healthy}; label 1/0.
   - Eksperimen B: rekaman berlabel ahli; label dari diagnosis_1..4 (COVID-19 vs healthy_cough).
     Jika beberapa ahli memberi label berbeda, pakai majority vote; jika seri, buang. Catat
     jumlah konflik.
   - Eksperimen C (opsional): status_SSL.
   - Tambahkan opsi di config.yaml: `healthy_to_covid_ratio` (default 3, null = pakai semua) untuk
     undersampling kelas healthy secara acak (seed 42) bila komputasi terbatas. Diskusikan di
     PROGRESS.md dampaknya terhadap metrik.
2. Buat split STRATIFIED train/val/test = 70/15/15 (seed 42) per uuid, simpan di
   `data/processed/splits_A.csv` (dan _B). Split ini FINAL dan dipakai semua tahap berikutnya.
3. Buat `src/preprocessing.py` dengan fungsi-fungsi terpisah dan terdokumentasi (docstring Indonesia):
   a. `convert_to_wav()` — dekode webm/ogg dengan ffmpeg → WAV mono, 16 kHz, PCM 16-bit
      (jelaskan di docstring kaitan dengan sampling & quantization: sampling rate 16 kHz,
      bit depth 16-bit → 65.536 level).
   b. `trim_silence()` — buang hening awal/akhir (librosa.effects.trim, top_db di config).
   c. `segment_coughs()` — segmentasi batuk berbasis energi (RMS/hysteresis threshold; boleh
      mengadaptasi pendekatan segmentasi dari repo resmi COUGHVID jika tersedia di dataset).
      Simpan metadata jumlah segmen per rekaman.
   d. `reduce_noise()` — opsional (noisereduce, spectral gating), bisa di-on/off di config.
   e. `normalize()` — peak normalization ke -1..1 (atau RMS), dan pad/potong ke durasi tetap
      (max_duration_s) untuk input CNN.
4. Buat `scripts/02_preprocess.py`: multiprocessing (joblib, n_jobs=-2), tqdm, resume (skip file
   yang sudah ada), `--limit N`, log error per file ke `reports/logs/preprocess_errors.csv`
   (file rusak/kosong/terlalu pendek < 0.3 s dibuang dan dicatat).
   Uji dulu dengan `--limit 50`, periksa hasilnya, baru jalankan penuh.
5. Hasil akhir: `data/processed/manifest_A.csv` (uuid, path_wav, label, split, durasi asli,
   durasi setelah trim, n_segmen, sr) dan manifest_B.csv.
6. Notebook `notebooks/02_preprocessing.ipynb` — untuk 1 contoh COVID-19 dan 1 contoh healthy,
   tampilkan before/after tiap langkah: waveform asli, setelah trim, hasil segmentasi (kotak segmen
   di atas waveform), setelah noise reduction & normalisasi. Tambahkan distribusi durasi
   sebelum/sesudah preprocessing per kelas. Simpan `reports/figures/02_*.png`. Eksekusi notebook.
7. Perbarui PROGRESS.md: jumlah data akhir per kelas per split (tabel), jumlah file gagal,
   parameter preprocessing yang dipakai, waktu proses. Berhenti dan beri ringkasan.
```
