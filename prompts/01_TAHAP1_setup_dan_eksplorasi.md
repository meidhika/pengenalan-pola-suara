# TAHAP 1 — Setup Proyek, Ekstraksi Dataset, dan Eksplorasi Metadata

> Tempel prompt di bawah ini di Claude Code (sesi baru). Pastikan `CLAUDE.md` dan folder
> `references/` sudah ada di root proyek (lihat PANDUAN.md).

---

```
Baca CLAUDE.md terlebih dahulu. Kita mengerjakan TAHAP 1 saja: setup proyek, ekstraksi dataset,
dan eksplorasi metadata COUGHVID. Jangan masuk ke preprocessing audio.

Langkah:
1. Cek lingkungan: versi Python, ketersediaan ffmpeg (`ffmpeg -version`), GPU (nvidia-smi bila ada),
   dan sisa ruang disk di drive D (butuh ±15 GB untuk ekstrak + wav). Laporkan bila ada yang kurang
   beserta cara instalnya di Windows (mis. `winget install Gyan.FFmpeg`), lalu tunggu konfirmasi saya
   jika ffmpeg belum terpasang.
2. Buat struktur folder sesuai CLAUDE.md bagian 3, `.gitignore` (abaikan data/, models/, .venv/,
   *.zip), dan virtual env `.venv`.
3. Buat `requirements.txt` (versi dipin seperlunya): numpy, pandas, scipy, matplotlib, seaborn,
   librosa, soundfile, pydub/ffmpeg-python, noisereduce, spafe (untuk GFCC), scikit-learn,
   imbalanced-learn, xgboost, joblib, pyyaml, tqdm, pyarrow, jupyter, nbconvert, ipykernel,
   torch, torchaudio, python-docx, python-pptx. Install ke .venv. Daftarkan kernel jupyter "ppkb".
4. Buat `config.yaml` + `src/config.py` (loader). Isi awal: path (root, raw, interim, processed,
   features, figures), seed=42, sample_rate=16000, cough_detected_threshold=0.8, max_duration_s=10.
5. Buat `scripts/01_extract.py`:
   - Inspeksi isi zip TANPA mengekstrak semua (`zipfile.namelist()`), ringkas jumlah file per ekstensi.
   - Jika ada zip bersarang (mis. public_dataset_v3.zip), tangani dengan benar.
   - Ekstrak ke `data/raw/` dengan progress bar, idempoten (skip bila sudah ada).
   - Simpan ringkasan ke `reports/logs/extract_summary.txt`.
6. Jalankan ekstraksi. Verifikasi: jumlah file audio (.webm/.ogg), .json, dan lokasi metadata_compiled.csv.
7. Buat `src/data_utils.py` dengan fungsi `load_metadata()` dan pemetaan uuid → path file audio.
   Cek berapa uuid di metadata yang file audionya ada.
8. Buat notebook `notebooks/01_eksplorasi_metadata.ipynb` (EDA), yang menampilkan:
   - Daftar kolom & tipe, persentase missing per kolom.
   - Distribusi `status`, `status_SSL`, `gender`, `age` (histogram), `respiratory_condition`,
     `fever_muscle_pain`.
   - Histogram `cough_detected` + jumlah rekaman yang lolos threshold 0.8.
   - Crosstab status × cough_detected≥0.8 (inilah populasi Eksperimen A).
   - Jumlah rekaman berlabel ahli (kolom diagnosis_1..4) dan distribusi nilai diagnosis-nya.
   - Sampel 5 rekaman per kelas: durasi & sample rate asli (pakai librosa/soundfile via ffmpeg).
   Simpan grafik ke `reports/figures/01_*.png` dan tabel ringkasan ke `reports/tables/01_*.csv`.
   Eksekusi notebook dengan nbconvert agar output tersimpan.
9. Salin konten bagian "Ringkasan referensi" CLAUDE.md menjadi `references/references.bib`
   (format BibTeX, 5 entri R0–R4).
10. Buat PROGRESS.md: status Tahap 1, angka-angka kunci (total rekaman, jumlah per status setelah
    filter, jumlah berlabel ahli, nilai unik diagnosis), dan rekomendasi strategi sampling untuk
    Tahap 2 (mis. pakai semua COVID-19 dan sampel healthy dengan rasio tertentu bila komputasi
    terbatas). Lalu berhenti dan beri ringkasan singkat.
```
