# PANDUAN PEMAKAIAN MASTER PROMPT — Proyek pola-suara (COUGHVID)

## Isi paket
| File | Fungsi |
|------|--------|
| `00_CLAUDE.md` | Konteks permanen proyek → **ganti nama jadi `CLAUDE.md`** dan taruh di root proyek |
| `01_TAHAP1_setup_dan_eksplorasi.md` | Setup env, ekstrak zip, EDA metadata |
| `02_TAHAP2_preprocessing.md` | Label, split, konversi wav 16 kHz, trim, segmentasi, normalisasi |
| `03_TAHAP3_ekstraksi_fitur.md` | Energy, ZCR, spektral, pitch, MFCC, GFCC, log-mel + visualisasi |
| `04_TAHAP4_model_ml_klasik.md` | Dummy, LR, SVM, k-NN, RF, XGBoost + ablasi fitur |
| `05_TAHAP5_model_cnn.md` | LightCNN (PyTorch) pada log-mel & MFCC 2D |
| `06_TAHAP6_evaluasi_dan_analisis.md` | Tabel final, uji signifikansi, label ahli vs self-reported, analisis error |
| `07_TAHAP7_laporan.md` | 7a draf per bab (Markdown) → 7b laporan Word/PDF |
| `08_TAHAP8_presentasi_dan_finalisasi.md` | Slide PPTX, demo prediksi, README, QA, folder submission |

## Persiapan (sekali saja, manual)
1. Pastikan folder proyek berisi zip dataset:
   `D:\Meidhika\Lecturer\S2\Semester 3\Pengenalan Pola dan Kecerdasan Buatan\Tugas\pola-suara\The COUGHVID crowdsourcing dataset.zip`
2. Salin `00_CLAUDE.md` ke folder tersebut dengan nama **`CLAUDE.md`**.
3. Buat subfolder `references\` dan salin 4 PDF artikel + PDF materi Pertemuan 8 ke sana.
4. (Disarankan) Pasang ffmpeg: `winget install Gyan.FFmpeg`, lalu buka terminal baru.
5. (Opsional) `git init` di folder proyek agar setiap tahap bisa di-commit dan mudah di-rollback.
6. Buka terminal di folder proyek, jalankan `claude`.

## Alur kerja per tahap
```
cd "D:\Meidhika\Lecturer\S2\Semester 3\Pengenalan Pola dan Kecerdasan Buatan\Tugas\pola-suara"
claude
> (tempel isi blok prompt Tahap 1)
  ... tunggu selesai, periksa hasil & PROGRESS.md ...
> /clear
> (tempel prompt Tahap 2)
  ... dan seterusnya
```
- Yang ditempel hanya isi di dalam blok ``` pada tiap file tahap.
- **Selalu `/clear` di antara tahap.** Kesinambungan dijaga oleh `CLAUDE.md` (dibaca otomatis)
  dan `PROGRESS.md` (log serah-terima yang ditulis Claude Code di akhir tiap tahap).
- Bila context menipis di tengah tahap: `/compact`, atau `/clear` lalu ketik
  *"Baca CLAUDE.md dan PROGRESS.md, lanjutkan Tahap X dari langkah Y."*
- Periksa hasil sebelum lanjut, terutama: jumlah data per kelas (Tahap 1–2) dan apakah angka
  AUC masuk akal (Tahap 4–5). AUC 0.60–0.75 untuk label self-reported itu wajar.

## Estimasi waktu komputasi (laptop biasa, CPU)
| Tahap | Perkiraan |
|-------|-----------|
| Ekstrak zip | 5–15 menit |
| Konversi + preprocessing (subset ±5–8 ribu file) | 20–60 menit |
| Ekstraksi fitur (pyin cukup lambat) | 30–90 menit |
| ML klasik + tuning | 10–40 menit |
| CNN (3 seed × 2 fitur) | 1–4 jam di CPU, jauh lebih cepat dengan GPU |
Rasio `healthy_to_covid_ratio` di `config.yaml` bisa diturunkan bila terlalu lama.

## Kustomisasi cepat
- Format sitasi APA → ubah di prompt 7a ("gaya IEEE (atau APA ...)").
- Format laporan dari dosen (font, margin, template kampus) → sebutkan di prompt 7b, atau taruh
  template .docx di `references/` dan minta dipakai sebagai reference.docx.
- Ingin pakai TensorFlow/Keras alih-alih PyTorch → ubah di CLAUDE.md bagian 2 dan prompt Tahap 5.
- Waktu mepet → lewati Eksperimen C, CRNN, Grad-CAM, dan demo Gradio (semuanya bertanda opsional).

## Checklist pengumpulan
- [ ] Laporan .docx dan .pdf (nama & NIM anggota sudah diisi)
- [ ] Slide presentasi .pptx
- [ ] Kode (src, scripts, notebooks sudah dieksekusi, config, README, requirements)
- [ ] Angka di laporan = angka di reports/tables/
- [ ] Sitasi dataset COUGHVID & 4 artikel referensi
