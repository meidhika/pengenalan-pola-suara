# Ringkasan Hasil — Klasifikasi Suara Batuk COVID-19 (COUGHVID v3)

Sumber angka: `reports/tables/06_*.csv` (dihasilkan `scripts/06_evaluate_all.py`). Semua metrik pada **split test**
yang dievaluasi sekali. Ambang keputusan = Youden's J dipilih di val.

## Temuan utama
1. **Eksperimen A (label self-reported, COVID-19 vs healthy) hanya sedikit di atas acak.** Test berisi 1.624 rekaman
   (108 COVID-19, prevalensi 0,067). ROC-AUC semua model berada di 0,54–0,58.
   - SVM RBF: **0,576 [CI 95% 0,523–0,629]**.
   - k-NN: 0,580, nilai test tertinggi.
   - CNN log-mel: 0,567, rata-rata 3 seed 0,572 ± 0,006.
   - CNN MFCC 2D: 0,544, rata-rata 3 seed 0,579 ± 0,030.
   - PR-AUC 0,08–0,10, hampir sama dengan prevalensi 0,067. Precision ≈ 0,08: dari 12 rekaman yang diprediksi
     COVID-19, hanya ±1 yang benar. Model **tidak layak untuk skrining**.
2. **Tidak ada model yang terbukti lebih baik.** Uji DeLong dan bootstrap berpasangan pada test yang sama:
   - k-NN vs SVM: Δ = +0,004, p = 0,87.
   - SVM vs CNN log-mel: p = 0,70.
   - SVM vs CNN MFCC 2D: p = 0,39.
   - SVM vs baseline *dummy* juga tidak bermakna (Δ = +0,060, p = 0,12), walaupun batas bawah CI AUC SVM (0,523) > 0,5.
   - Kriteria seleksi (AUC val) memilih CNN MFCC 2D seed 42, padahal AUC test-nya justru terendah. Artinya peringkat
     model di A tidak stabil.
3. **CNN ringan tidak mengungguli ML klasik.** Ini terjadi meskipun CNN memakai augmentasi (SpecAugment, *random crop*)
   dan memiliki 97 ribu parameter. Representasi (MFCC vs log-mel vs GFCC) juga hanya berpengaruh kecil.
4. **Eksperimen B (label ahli/pulmonolog) jauh lebih baik dengan fitur & pipeline yang sama.** SVM mencapai
   **0,826 [0,745–0,888]** (vs dummy: Δ = +0,339, p < 0,001). Kualitas label adalah faktor pembatas utama
   (sejalan dengan R4).
5. **Label self-reported dan label ahli praktis tidak sepakat.** Pada 572 rekaman yang memiliki kedua label:
   - kesepakatan 50,3%, **Cohen's κ = 0,00**;
   - hanya 43% rekaman berstatus "COVID-19" (self-reported) juga dinilai COVID-19 oleh ahli.
6. **Uji silang (uuid training/validasi model sumber dibuang dari set evaluasi):**
   - **S1, test B dengan label ahli (n = 102):** model B mencapai AUC 0,84–0,90. Model ML A (dilatih dengan
     label self-reported) tetap mencapai 0,61–0,71 (SVM 0,673, RF 0,711) dan CNN log-mel 0,653, tetapi CNN MFCC 2D
     hanya 0,489. Jadi model A menangkap
     sebagian pola akustik yang juga dinilai ahli.
   - **S2, test A dengan label self-reported (n = 1.554):** model B hanya mencapai 0,55–0,58, setara atau sedikit di
     bawah model A (0,57–0,59). Label self-reported tidak dapat diprediksi dengan baik bahkan oleh model yang lebih "bersih".
   - **S3, prediksi model A yang sama dinilai dengan dua label (n = 251):** AUC terhadap label ahli 0,52–0,65,
     terhadap label self-reported 0,49–0,55.
   - *Caveat*: ahli mendiagnosis dengan **mendengar rekaman yang sama**, sehingga label ahli "konsisten secara akustik"
     secara konstruksi dan bukan *ground truth* PCR. AUC tinggi di B mengukur kesesuaian dengan penilaian akustik
     pulmonolog, bukan akurasi diagnosis COVID-19.
7. **Analisis error (SVM A):**
   - Skor sangat sempit dan tumpang-tindih antar-kelas: median healthy 0,242 vs COVID-19 0,254. Kalibrasi
     probabilitas rendah, sehingga ambang dari val wajib dipakai.
   - FP didominasi rekaman panjang dengan banyak batuk (median 5,0 s vs TN 3,0 s). FN sering berupa rekaman sangat
     pendek (0,8–1,0 s) atau dengan sedikit batuk.
   - Skor berkorelasi dengan durasi: Spearman ρ = 0,41 (SVM) dan 0,61 (CNN log-mel). Padahal durasi saja hanya
     memberi AUC 0,52. Model sebagian merespons **kepadatan/panjang batuk**, bukan pola spesifik COVID-19.
8. **Subgrup (fairness):**
   - AUC SVM lebih rendah pada subgrup bergejala: demam/nyeri otot "ya" 0,51 vs "tidak" 0,59; kondisi pernapasan
     "ya" 0,53 vs "tidak" 0,59.
   - Model tidak mampu membedakan COVID-19 dari batuk bergejala lain. Prevalensi COVID-19 di subgrup demam 0,23 vs
     0,05, sehingga gejala ber-*confound* dengan label.
   - Usia ≥ 55 tahun: AUC 0,48. Gender relatif setara (female 0,59, male 0,57).
   - CI subgrup lebar (12–50 positif per subgrup); perbedaan ini indikatif, belum konklusif.
9. **Dibanding literatur:** R2 dan R3 melaporkan accuracy 91–95% pada dataset Cambridge "COVID-19 Sounds" (multimodal,
   tidak per subjek, augmentasi tanpa urutan split yang jelas, tanpa AUC). R4 (call-centre, label PCR) melaporkan AUC RF
   0,83 (PCR) vs 0,79 (self-reported). Tidak ada yang memakai COUGHVID, jadi angka **tidak dapat dibandingkan langsung**.
   Hasil A kami (AUC ≈ 0,58 pada prevalensi alami 1:14) konsisten dengan kesimpulan R4 bahwa label crowdsourced
   kurang andal.

## Angka kunci
| Setting | Model | ROC-AUC test [CI 95%] | PR-AUC | Sens / Spes |
|---|---|---|---|---|
| A (self-reported), n = 1.624 | SVM RBF | 0,576 [0,523–0,629] | 0,088 | 0,556 / 0,553 |
| A | CNN log-mel (3 seed) | 0,572 ± 0,006 | 0,081 | 0,370 / 0,708 (seed 42) |
| A | Dummy | 0,516 [0,472–0,563] | 0,069 | – |
| B (ahli), n = 152 | SVM RBF + SMOTE | 0,826 [0,745–0,888] | 0,786 | 0,743 / 0,805 |
| Kesepakatan label (n = 572) | – | κ = 0,00 | – | – |

## Keterbatasan
- **Label self-reported** tanpa konfirmasi PCR, dan label ahli berbasis pendengaran (bukan PCR). Kesepakatan antar-ahli
  juga rendah (124 dari 130 rekaman yang dinilai >1 ahli mengalami konflik).
- **Tidak ada ID subjek.** Split dilakukan per rekaman (uuid), sehingga satu orang bisa muncul di train dan test.
- **Variasi mikrofon/perangkat dan lingkungan** (crowdsourced, web/smartphone, format webm/ogg/wav). Normalisasi peak
  menghilangkan informasi kenyaringan absolut.
- **Bias demografi:** 66% pria (test A); usia/gender sebagian tidak diisi; subgrup lansia & "other" kecil.
- **Undersampling train A 3:1** membuang 5.585 rekaman healthy dan menggeser kalibrasi probabilitas.
- Segmentasi batuk tidak sempurna (11,6% rekaman A tanpa segmen terdeteksi). Fitur dihitung dari sinyal penuh hasil
  *trim*, sehingga durasi/kepadatan batuk ikut terekam fitur.
- Test B kecil (152; uji silang S1 hanya 102), sehingga CI lebar. Repeated CV di train bukan *nested CV*.

## Saran pengembangan
1. Gunakan data berlabel PCR (Coswara, COVID-19 Sounds) untuk validasi eksternal lintas dataset.
2. Pakai label `status_SSL` (Eksperimen C) atau pelatihan *noise-robust* (label smoothing, *co-teaching*) untuk label noisy.
3. Gunakan fitur per segmen batuk (agregasi per batuk) agar model tidak bergantung pada durasi/jumlah batuk. Pertimbangkan
   *embedding* pra-latih (VGGish/PANNs/wav2vec 2.0) atau openSMILE ComParE seperti R4.
4. Kalibrasi probabilitas (Platt/isotonic di val) dan pilih ambang berbasis target sensitivitas skrining.
5. Gabungkan metadata gejala (demam, kondisi pernapasan) secara eksplisit dan laporkan apakah suara menambah informasi
   di atas gejala.
6. Uji augmentasi level waveform (R2), kernel 1×12 / CRNN, dan *ensemble* seed (log-mel 0,576; MFCC 0,591) tanpa
   memilih berdasarkan test.
