# TAHAP 6 — Evaluasi Komprehensif, Perbandingan, dan Analisis

> Ketik `/clear` dulu.

---

```
Baca CLAUDE.md dan PROGRESS.md. Kita mengerjakan TAHAP 6 saja: konsolidasi hasil, analisis, dan
menyiapkan semua tabel/figur final untuk laporan. Jangan melatih ulang model kecuali benar-benar
perlu (gunakan model tersimpan di models/).

Langkah:
1. `scripts/06_evaluate_all.py` + `notebooks/06_evaluasi_akhir.ipynb`:
   a. Tabel final semua model (Eksperimen A): Accuracy, Balanced Acc, Precision, Recall,
      Specificity, F1, ROC-AUC [95% CI], PR-AUC → `reports/tables/06_tabel_final_A.csv` dan versi
      Markdown. Tandai model terbaik.
   b. Uji signifikansi perbedaan AUC dua model teratas (DeLong atau bootstrap berpasangan).
   c. Satu plot ROC gabungan model terbaik ML vs CNN vs baseline.
   d. Eksperimen B vs A (terinspirasi R4): bandingkan performa model yang dilatih dengan label
      ahli vs label self-reported; jika memungkinkan, uji silang (latih di A, uji di subset
      berlabel ahli yang tidak dipakai saat training). Diskusikan keandalan label crowdsourced.
   e. Analisis error: distribusi probabilitas prediksi per kelas, contoh false positive / false
      negative (tampilkan spectrogram 3 contoh masing-masing), dan performa per subgrup
      (gender, kelompok umur, respiratory_condition, fever_muscle_pain) → tabel fairness/subgroup.
   f. Tabel perbandingan dengan literatur (R2, R3, R4, dan paper COUGHVID R0): dataset, fitur,
      model, metrik yang dilaporkan, vs hasil kita — dengan catatan perbedaan setting
      (label, jumlah data, split, augmentasi sebelum split, dsb.). Jangan menyalin angka yang
      tidak ada di referensi; jika tidak yakin dengan angka sebuah paper, tulis "lihat paper"
      dan beri tanda TODO untuk saya cek manual.
2. Buat `reports/ringkasan_hasil.md` (≤ 1,5 halaman): temuan utama dalam poin-poin, angka kunci,
   keterbatasan (label self-reported, tidak ada ID subjek, variasi mikrofon, bias demografi,
   undersampling), dan saran pengembangan. File ini akan menjadi bahan utama laporan di Tahap 7.
3. Daftar semua figur & tabel final beserta caption Bahasa Indonesia di
   `reports/daftar_figur_tabel.md` (nama file → nomor Gambar/Tabel → caption).
4. Perbarui PROGRESS.md. Berhenti dan ringkas.
```
