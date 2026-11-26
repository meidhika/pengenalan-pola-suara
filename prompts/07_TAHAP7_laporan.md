# TAHAP 7 — Penyusunan Laporan Ilmiah

> Ketik `/clear` dulu. Tahap ini sengaja dipecah menjadi 7a (draf Markdown per bab) dan 7b (konversi ke
> Word) agar context tidak penuh. Jika context mulai penuh di 7a, jalankan `/compact` atau `/clear`
> lalu lanjutkan dengan "Lanjutkan Tahap 7a mulai Bab X".

---

## 7a — Draf laporan per bab (Markdown)

```
Baca CLAUDE.md, PROGRESS.md, reports/ringkasan_hasil.md, dan reports/daftar_figur_tabel.md.
Kita mengerjakan TAHAP 7a: menulis draf laporan dalam Bahasa Indonesia baku-akademik, satu file
Markdown per bab di `reports/final/draft/`. Ambil angka HANYA dari file di reports/tables/
(jangan mengarang angka). Rujuk gambar dengan path relatif ke reports/figures/.

Judul (boleh disempurnakan): "Klasifikasi Suara Batuk untuk Skrining COVID-19 Menggunakan
Ekstraksi Fitur MFCC–GFCC dan Perbandingan Machine Learning Klasik dengan CNN Ringan pada
Dataset COUGHVID".

Struktur:
- 00_abstrak.md — Abstrak (≤ 250 kata) + kata kunci (Bahasa Indonesia; tambahkan Abstract
  bahasa Inggris).
- 01_pendahuluan.md — latar belakang (kebutuhan skrining murah & cepat, PCR invasif), rumusan
  masalah, tujuan, manfaat, batasan masalah.
- 02_tinjauan_pustaka.md — (a) dasar teori suara sesuai materi kuliah: gelombang, amplitudo,
  frekuensi, periode, pitch, timbre, periodik vs nonperiodik, analog-digital, sampling &
  quantization, waveform, domain waktu vs frekuensi, Fourier/FFT, STFT & spectrogram;
  (b) fitur: energy, ZCR, fitur spektral, MFCC (tahapan: pre-emphasis, framing, windowing, FFT,
  mel filterbank, log, DCT — sertakan rumus mel = 2595·log10(1+f/700)), GFCC; (c) klasifier:
  SVM, RF, XGBoost, CNN; (d) penelitian terkait R0–R4 dalam tabel (penulis, tahun, dataset, metode,
  hasil, keterbatasan).
- 03_metodologi.md — diagram alur penelitian mengikuti pipeline slide 10 (akuisisi →
  preprocessing → ekstraksi fitur → klasifikasi → evaluasi; buat diagramnya sebagai PNG dengan
  matplotlib/graphviz ke reports/figures/07_alur_penelitian.png), deskripsi dataset, kriteria
  inklusi, skema label A/B, split, detail preprocessing, fitur, model & hyperparameter,
  penanganan imbalance, metrik evaluasi (rumus), lingkungan komputasi.
- 04_hasil_dan_pembahasan.md — EDA, visualisasi representasi suara, hasil ML klasik, ablasi
  fitur, hasil CNN, perbandingan final, eksperimen keandalan label, analisis error & subgrup,
  perbandingan dengan literatur, pembahasan (mengapa hasil demikian, implikasi praktis).
- 05_kesimpulan.md — kesimpulan menjawab tiap rumusan masalah + saran.
- 06_daftar_pustaka.md — gaya IEEE (atau APA, ikuti yang saya minta), dari references/references.bib,
  plus referensi pustaka librosa/scikit-learn/PyTorch.
- 07_lampiran.md — struktur repo, cara menjalankan, tabel hyperparameter lengkap, pembagian
  tugas anggota kelompok (biarkan placeholder [NAMA ANGGOTA] untuk saya isi).

Sitasi dalam teks memakai format [1], [2], ... Setiap gambar/tabel punya nomor & caption dan
dirujuk di teks. Tulis bab demi bab; setelah tiap bab selesai, catat di PROGRESS.md bab mana
yang sudah selesai, agar bisa dilanjutkan bila sesi terputus.
```

## 7b — Konversi ke dokumen akhir

```
Baca CLAUDE.md dan PROGRESS.md. TAHAP 7b: gabungkan semua bab di reports/final/draft/ menjadi
laporan Word `reports/final/Laporan_COUGHVID_Kelompok.docx` menggunakan python-docx (atau pandoc
jika terpasang, dengan reference.docx berformat: Times New Roman 12, spasi 1.5, margin 3-3-3-3 cm
atau sesuai instruksi saya). Ketentuan:
- Halaman sampul (judul, mata kuliah, dosen pengampu: Enny Itje Sela, placeholder nama & NIM
  anggota, program studi Magister Teknologi Informasi, tahun 2026).
- Daftar isi otomatis (field TOC), daftar gambar, daftar tabel.
- Gambar disisipkan dengan lebar ≤ 15 cm, caption di bawah gambar; caption tabel di atas tabel.
- Tabel dari CSV dirender sebagai tabel Word (bukan gambar), angka 3–4 desimal.
- Ekspor juga ke PDF jika LibreOffice/Word tersedia (docx2pdf atau soffice --headless).
Setelah itu buka/cek dokumen: jumlah halaman, semua gambar muncul, tidak ada placeholder yang
terlewat kecuali [NAMA ANGGOTA]. Laporkan daftar hal yang perlu saya isi/cek manual.
Perbarui PROGRESS.md dan berhenti.
```
