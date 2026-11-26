# TAHAP 8 — Slide Presentasi, Demo, dan Finalisasi Repo

> Ketik `/clear` dulu.

---

```
Baca CLAUDE.md, PROGRESS.md, dan reports/ringkasan_hasil.md. TAHAP 8 (terakhir):

1. Slide presentasi `reports/final/Presentasi_COUGHVID_Kelompok.pptx` (python-pptx), 12–15 slide,
   16:9, desain bersih dan konsisten, teks ringkas (≤ 6 poin per slide), memakai figur dari
   reports/figures/:
   Judul & anggota → Latar belakang → Rumusan masalah & tujuan → Dataset COUGHVID →
   Alur penelitian (pipeline slide 10) → Preprocessing (before/after) → Representasi suara
   (waveform, spectrogram, MFCC, GFCC) → Fitur & uji statistik → Model (ML klasik + LightCNN) →
   Hasil (tabel final + ROC) → Keandalan label (Eksp. B) → Analisis error →
   Perbandingan literatur → Kesimpulan & saran → Terima kasih/Q&A.
   Tambahkan speaker notes Bahasa Indonesia di tiap slide (±60–90 detik bicara).
2. Demo inferensi `scripts/predict.py`: input satu file audio (wav/webm/ogg/mp3) → jalankan
   preprocessing + fitur yang sama → output probabilitas & label dari model terbaik, dengan
   disclaimer bahwa ini bukan alat diagnosis medis. (Opsional: `app_demo.py` dengan Gradio untuk
   rekam/upload batuk.)
3. README.md lengkap: deskripsi proyek, struktur folder, cara instal (.venv + requirements),
   urutan menjalankan script 01→06, cara reproduksi, ringkasan hasil, lisensi dataset (CC BY 4.0)
   dan sitasi COUGHVID.
4. Pemeriksaan akhir (QA):
   - Jalankan ulang semua notebook dari awal (nbconvert --execute) dan pastikan tanpa error.
   - Cek konsistensi angka: angka di laporan .docx & slide = angka di reports/tables/.
   - Cek tidak ada data leakage (split dipakai konsisten; scaler/SMOTE/augmentasi hanya di train).
   - `pip freeze > requirements-lock.txt`.
   - Buat `submission/` berisi: laporan (.docx & .pdf), slide (.pptx), dan zip kode
     (src, scripts, notebooks, config, README, requirements) TANPA data/ dan .venv.
5. Perbarui PROGRESS.md menjadi status "SELESAI" + checklist pengumpulan. Berikan ringkasan akhir
   dan daftar hal yang masih harus saya isi/cek manual.
```
