# BAB IV HASIL DAN PEMBAHASAN

Kecuali disebutkan lain, semua metrik pada bab ini dihitung pada **split test yang dievaluasi satu kali**. Ambang
keputusan memakai indeks Youden yang dipilih di val. Angka dalam kurung siku menyatakan interval kepercayaan 95%
(*bootstrap*).

## 4.1 Analisis Data Eksploratif (EDA)

### 4.1.1 Karakteristik rekaman asli

Gambar 6 menunjukkan contoh *waveform* rekaman asli untuk ketiga status *self-reported*. Format dan *sampling rate*
rekaman beragam (Opus 48 kHz, PCM/AAC 16–48 kHz), sehingga konversi seragam ke 16 kHz mono PCM 16-bit pada tahap
*preprocessing* diperlukan. Setelah filter `cough_detected` ≥ 0,8, populasi Eksperimen A sangat tidak seimbang,
yaitu 720 COVID-19 berbanding 10.132 *healthy* (±1:14).

![Gambar 6. Contoh waveform rekaman batuk asli untuk tiap status self-reported beserta codec, sampling rate asli, dan durasinya.](../../figures/01_contoh_waveform.png)

**Gambar 6.** Contoh *waveform* (amplitudo terhadap waktu) rekaman batuk asli untuk tiap status *self-reported*,
beserta codec, *sampling rate* asli, dan durasinya.

### 4.1.2 Durasi

Durasi asli rekaman hampir sama antar-kelas (median ±9,7–9,8 s), karena aplikasi perekam membatasi durasi ±10 s.
Perbedaan baru muncul setelah *trimming* (Tabel 8, Gambar 7). Pada Eksperimen A selisihnya kecil (4,40 vs 3,98 s).
Pada Eksperimen B selisihnya cukup besar: rekaman berlabel ahli COVID-19 lebih panjang (4,92 s) dan memuat lebih
banyak batuk (1,75 s) dibanding *healthy cough* (3,17 s dan 0,99 s). Durasi karena itu berpotensi menjadi jalan
pintas (*shortcut*) bagi model, dan hal ini diperiksa melalui ablasi (Subbab 4.4).

**Tabel 8.** Median durasi rekaman (detik) per kelas.

| Eksperimen | Kelas | n | Asli | Setelah *trim* | Total segmen batuk |
|---|---|---|---|---|---|
| A | COVID-19 | 718 | 9,78 | 4,40 | 1,43 |
| A | *healthy* | 4.537 | 9,72 | 3,98 | 1,30 |
| B | COVID-19 | 470 | 9,78 | 4,92 | 1,75 |
| B | *healthy* | 541 | 9,60 | 3,17 | 0,99 |

<!-- sumber: reports/tables/02_ringkasan_durasi.csv (kolom 50%) -->

![Gambar 7. Perbandingan durasi setelah trim per kelas pada Eksperimen A dan B.](../../figures/02_boxplot_durasi.png)

**Gambar 7.** Perbandingan durasi setelah *trim* per kelas pada Eksperimen A dan B.

## 4.2 Visualisasi Representasi Suara

Gambar 8–10 memperlihatkan satu rekaman dalam berbagai representasi yang dibahas pada Bab II.

- **Domain waktu (Gambar 8):** *waveform* batuk terdiri atas letupan-letupan singkat beramplitudo tinggi yang
  diikuti peluruhan mirip derau. Tidak ada pola berulang yang teratur, sehingga batuk bersifat **nonperiodik**.
  Secara visual, *waveform* COVID-19 dan *healthy* tidak dapat dibedakan.
- **Domain frekuensi (Gambar 9):** spektrum FFT satu *frame* 32 ms memperlihatkan energi yang
  tersebar lebar dari ratusan Hz hingga ±8 kHz, tanpa deret harmonik yang jelas. Pola ini merupakan ciri bunyi
  nonperiodik dan sejalan dengan rendahnya rasio *frame voiced*: 14% rekaman A tidak memiliki *frame* bersuara sama
  sekali.
- **Waktu–frekuensi (Gambar 10):** *spectrogram* STFT memperlihatkan setiap batuk sebagai "pita vertikal"
  berenergi lebar. *Log-mel* memadatkan sumbu frekuensi sesuai skala mel. MFCC dan GFCC meringkas selubung
  spektrum menjadi beberapa koefisien yang menjadi dasar fitur tabular.

![Gambar 8. Waveform batuk COVID-19 vs healthy setelah preprocessing (domain waktu).](../../figures/03_waveform_covid_vs_healthy.png)

**Gambar 8.** *Waveform* batuk COVID-19 vs *healthy* setelah *preprocessing* (domain waktu; batuk bersifat
nonperiodik).

![Gambar 9. Spektrum magnitudo FFT satu frame 32 ms (domain frekuensi) untuk batuk COVID-19 dan healthy.](../../figures/03_spektrum_fft_frame.png)

**Gambar 9.** Spektrum magnitudo FFT satu *frame* 32 ms (domain frekuensi) untuk batuk COVID-19 dan *healthy*.

![Gambar 10. Representasi waktu–frekuensi satu rekaman: spektrogram STFT, log-mel, MFCC, dan GFCC.](../../figures/03_representasi_waktu_frekuensi.png)

**Gambar 10.** Representasi waktu–frekuensi satu rekaman: *spectrogram* STFT, *log-mel*, MFCC, dan GFCC.

### 4.2.1 Daya beda fitur (train A)

Uji Mann–Whitney dengan koreksi FDR pada 315 fitur (train A) memberi hasil berikut (Tabel 9).

- Hanya **39 dari 315 fitur** berbeda signifikan antara kelas COVID-19 dan *healthy*.
- Tidak satu pun fitur mencapai efek sedang ($|r| \ge 0{,}28$). Fitur terkuat adalah `contrast_05_std`, yaitu
  variasi kontras spektral pada pita tertinggi, dengan AUC univariat hanya 0,584 ($r$ = 0,17).
- Kelompok spektral paling informatif (18/40 fitur signifikan), sedangkan *pitch* sama sekali tidak signifikan.
- Fitur "klasik" seperti ZCR, RMS rata-rata, *centroid*, *rolloff*, dan $f_0$ berada pada AUC 0,48–0,52, yang
  praktis setara dengan tebakan acak.

**Tabel 9.** Ringkasan uji beda fitur per kelompok (Mann–Whitney, FDR < 0,05; train Eksperimen A).

| Kelompok | Jumlah fitur | Signifikan | $\lvert r \rvert$ maks. | Fitur terkuat |
|---|---|---|---|---|
| Spektral | 40 | 18 | 0,168 | `contrast_05_std` |
| Waktu | 9 | 3 | 0,122 | `rms_min` |
| MFCC Δ | 52 | 6 | 0,121 | `mfcc_d_00_min` |
| MFCC | 78 | 4 | 0,119 | `mfcc_08_min` |
| GFCC | 78 | 5 | 0,100 | `gfcc_00_min` |
| MFCC ΔΔ | 52 | 3 | 0,096 | `mfcc_dd_04_min` |
| *Pitch* | 6 | 0 | 0,072 | `f0_std` |

<!-- sumber: reports/tables/03_uji_fitur_per_kelompok.csv -->

Rata-rata *log-mel* per kelas (Gambar 11) hanya berselisih 0,4–1,6 dB per pita. Proyeksi PCA dan t-SNE (Gambar 12)
memperlihatkan kedua kelas bercampur total: nilai *silhouette* dengan label sebagai klaster ≈ 0 (−0,006 di ruang
fitur penuh). Sebaliknya, pada train Eksperimen B (label ahli) **178 dari 315 fitur** signifikan dengan efek sedang
($|r|$ ≈ 0,27–0,33). Fitur terkuat di B adalah `durasi_s` (AUC 0,667). Temuan ini menjadi indikasi awal bahwa sumber
label sangat menentukan keterpisahan kelas.

![Gambar 11. Rata-rata log-mel spectrogram per kelas dan selisihnya (COVID-19 − healthy).](../../figures/03_rata2_logmel_per_kelas.png)

**Gambar 11.** Rata-rata *log-mel spectrogram* per kelas dan selisihnya (COVID-19 − *healthy*), train Eksperimen A.

![Gambar 12. Proyeksi PCA dan t-SNE fitur tabular; kedua kelas sangat tumpang-tindih.](../../figures/03_proyeksi_pca_tsne.png)

**Gambar 12.** Proyeksi PCA dan t-SNE fitur tabular train Eksperimen A; kedua kelas sangat tumpang-tindih.

## 4.3 Hasil Model ML Klasik

### 4.3.1 Eksperimen A (label *self-reported*)

Tabel 10 dan Gambar 13 menyajikan kinerja semua model pada test A (n = 1.624; 108 COVID-19; prevalensi 0,067).
Seluruh model ML klasik hanya sedikit di atas acak, dengan **ROC-AUC 0,569–0,580**. Hasil ini konsisten dengan AUC CV
di train (0,564–0,583) dan AUC val (0,527–0,574). SVM RBF terpilih sebagai ML terbaik menurut AUC val (0,574). SVM
mencapai AUC test 0,576 [0,523–0,629], sensitivitas 0,556, dan spesifisitas 0,553. PR-AUC semua model berada pada
0,087–0,097, hanya sedikit di atas prevalensi 0,067. *Precision* ±0,08 berarti dari setiap ±12 rekaman yang diprediksi
COVID-19, hanya ±1 yang benar.

**Tabel 10.** Kinerja semua model Eksperimen A pada test (n = 1.624; prevalensi 0,067). ★ = AUC val tertinggi
(kriteria seleksi yang ditetapkan sebelumnya), ▲ = AUC test tertinggi (informasi).

| Model | Bal. Acc. | Precision | Sens. | Spes. | F1 | ROC-AUC [CI 95%] | PR-AUC | AUC val |
|---|---|---|---|---|---|---|---|---|
| *Dummy* (stratified) | 0,500 | 0,067 | 1,000 | 0,000 | 0,125 | 0,516 [0,472–0,563] | 0,069 | 0,476 |
| *Logistic Regression* | 0,541 | 0,079 | 0,481 | 0,600 | 0,136 | 0,569 [0,514–0,623] | 0,087 | 0,561 |
| SVM (RBF) | 0,554 | 0,081 | 0,556 | 0,553 | 0,142 | 0,576 [0,523–0,629] | 0,088 | 0,574 |
| k-NN ▲ | 0,561 | 0,086 | 0,509 | 0,613 | 0,147 | 0,580 [0,525–0,636] | 0,088 | 0,542 |
| *Random Forest* | 0,556 | 0,090 | 0,407 | 0,705 | 0,147 | 0,570 [0,513–0,628] | 0,091 | 0,527 |
| XGBoost | 0,531 | 0,073 | 0,630 | 0,433 | 0,131 | 0,572 [0,513–0,629] | 0,097 | 0,546 |
| CNN (*log-mel*) | 0,539 | 0,083 | 0,370 | 0,708 | 0,136 | 0,567 [0,511–0,619] | 0,081 | 0,566 |
| CNN (MFCC 2D) ★ | 0,530 | 0,076 | 0,463 | 0,596 | 0,130 | 0,544 [0,487–0,601] | 0,086 | 0,580 |

<!-- sumber: reports/tables/06_tabel_final_A.csv (accuracy & ambang tersedia di CSV) -->

![Gambar 13. Kurva ROC dan Precision–Recall model ML klasik, Eksperimen A (test).](../../figures/04_roc_pr_A.png)

**Gambar 13.** Kurva ROC dan *Precision–Recall* model ML klasik, Eksperimen A (test).

Dampak *undersampling* train terlihat jelas pada kalibrasi model. Pada ambang 0,5, SVM A memprediksi **seluruh**
rekaman test sebagai *healthy* (sensitivitas 0). Ambang Youden dari val (0,249) diperlukan untuk memperoleh
keseimbangan sensitivitas–spesifisitas. Perbandingan strategi ketidakseimbangan (*class weight* vs SMOTE) hanya
menghasilkan selisih AUC CV ≤ 0,01. Analisis kepentingan fitur (Gambar 14) konsisten antar-metode (RF, XGBoost, dan
*permutation importance* SVM di val). Peringkat pertama selalu `contrast_05_max/std`, yaitu kontras spektral pita
tertinggi, disusul statistik MFCC koefisien tengah–tinggi (`mfcc_06_mean`, `mfcc_11_mean`, `mfcc_09_max`) dan
`mfcc_d_00_min`. Urutan ini sama dengan temuan uji univariat pada Subbab 4.2.1.

![Gambar 14. Fitur terpenting Eksperimen A (RF, XGBoost, permutation importance SVM).](../../figures/04_importance_A.png)

**Gambar 14.** Fitur terpenting Eksperimen A: *mean decrease impurity* RF, *gain* XGBoost, dan *permutation
importance* SVM pada val.

### 4.3.2 Eksperimen B (label ahli)

Dengan fitur dan *pipeline* yang **identik**, kinerja pada Eksperimen B jauh lebih tinggi (Tabel 11, Gambar 15).
SVM RBF + SMOTE menjadi model terbaik menurut val maupun test, dengan **ROC-AUC 0,826 [0,745–0,888]**,
sensitivitas 0,743, dan spesifisitas 0,805. Model lain berada pada AUC 0,776–0,817. Pada B, kepentingan fitur
didominasi statistik bentuk distribusi MFCC (`mfcc_06_skew`, `mfcc_08_mean`, `mfcc_06_kurt`), dinamika Δ/ΔΔ, serta
fitur *pitch* (`f0_max`, `voiced_ratio`) pada RF/XGBoost.

**Tabel 11.** Kinerja model ML klasik Eksperimen B (label ahli) pada test (n = 152; prevalensi 0,461).

| Model | Bal. Acc. | Precision | Sens. | Spes. | F1 | ROC-AUC [CI 95%] | PR-AUC | AUC val |
|---|---|---|---|---|---|---|---|---|
| *Dummy* (stratified) | 0,500 | 0,461 | 1,000 | 0,000 | 0,631 | 0,487 [0,417–0,567] | 0,454 | 0,428 |
| *Logistic Regression* | 0,794 | 0,783 | 0,771 | 0,817 | 0,777 | 0,812 [0,730–0,877] | 0,749 | 0,712 |
| SVM (RBF) ★ ▲ | 0,774 | 0,765 | 0,743 | 0,805 | 0,754 | 0,826 [0,745–0,888] | 0,786 | 0,718 |
| k-NN | 0,692 | 0,688 | 0,629 | 0,756 | 0,657 | 0,776 [0,689–0,848] | 0,675 | 0,648 |
| *Random Forest* | 0,766 | 0,718 | 0,800 | 0,732 | 0,757 | 0,817 [0,741–0,879] | 0,771 | 0,685 |
| XGBoost | 0,716 | 0,741 | 0,614 | 0,817 | 0,672 | 0,807 [0,726–0,871] | 0,754 | 0,713 |

<!-- sumber: reports/tables/06_tabel_final_B.csv -->

![Gambar 15. Kurva ROC dan Precision–Recall model ML klasik, Eksperimen B (test, label ahli).](../../figures/04_roc_pr_B.png)

**Gambar 15.** Kurva ROC dan *Precision–Recall* model ML klasik, Eksperimen B (test, label ahli).

AUC test B (0,78–0,83) sedikit lebih tinggi daripada AUC val (0,65–0,72) dan CV (0,72–0,77). Perbedaan ini wajar
karena ukuran sampelnya kecil: test B hanya berisi 152 rekaman, dan lebar CI-nya ±0,07.

## 4.4 Ablasi Kelompok Fitur

Ablasi dilakukan pada SVM. Setiap subset fitur di-*tuning* ulang, dan hasilnya disajikan pada Tabel 12 serta
Gambar 16.

**Tabel 12.** Ablasi kelompok fitur pada SVM (ROC-AUC test).

| Subset fitur | Jumlah | AUC test A | AUC test B |
|---|---|---|---|
| (i) Waktu (RMS, ZCR) | 8 | 0,485 | 0,680 |
| (ii) Spektral | 40 | 0,542 | 0,667 |
| (iii) MFCC (+Δ, ΔΔ) | 182 | 0,568 | 0,813 |
| (iv) GFCC | 78 | 0,495 | 0,764 |
| (v) MFCC + GFCC | 260 | 0,493 | 0,819 |
| (vi) Semua | 315 | 0,576 | 0,826 |
| (vii) Durasi saja | 1 | 0,518 | 0,671 |
| (viii) Semua tanpa durasi | 314 | 0,576 | 0,823 |

<!-- sumber: reports/tables/04_ablasi_fitur.csv, 04_ablasi_fitur_B.csv (AUC CV & val tersedia di CSV) -->

![Gambar 16. Ablasi kelompok fitur pada SVM, Eksperimen A.](../../figures/04_ablasi_A.png)

**Gambar 16.** Ablasi kelompok fitur (waktu, spektral, MFCC, GFCC, gabungan) pada SVM, Eksperimen A.

Ada empat temuan dari ablasi ini.

1. **MFCC adalah kelompok fitur tunggal terbaik** di kedua eksperimen (A 0,568; B 0,813).
2. **GFCC tidak menambah informasi di atas MFCC.** Pada B, MFCC + GFCC (0,819) ≈ MFCC saja (0,813). Pada A, GFCC
   bahkan tidak lebih baik dari acak (0,495). Hasil ini menunjukkan redundansi antara kedua representasi cepstral
   tersebut.
3. **Fitur domain waktu (RMS, ZCR) paling lemah**, yaitu tidak lebih baik dari acak pada A (0,485).
4. **Durasi bukan jalan pintas bagi model penuh.** Membuang durasi tidak mengubah AUC (A 0,576 → 0,576; B 0,826 →
   0,823). Namun, di B durasi saja sudah mencapai AUC 0,671, sehingga durasi tetap perlu dicatat sebagai
   *confounder* potensial pada label ahli.

Pada A, nilai test beberapa subset (GFCC, MFCC + GFCC) berada di bawah 0,5 walaupun AUC CV-nya ±0,56–0,58. Gejala
ini menunjukkan bahwa sinyal pada label *self-reported* sangat lemah, sehingga peringkat antar-subset tidak stabil.

## 4.5 Hasil CNN Ringan

Kurva pelatihan (Gambar 17) menunjukkan bahwa AUC val CNN *log-mel* mendatar di ±0,56 sejak ±*epoch* 10, sehingga
*early stopping* berhenti pada *epoch* 19–47. Selisih AUC train (teraugmentasi) dan val pada *checkpoint* terbaik
hanya 0,02–0,06, sehingga *overfitting* tergolong ringan. Pada CNN MFCC 2D, *overfitting* lebih jelas: AUC train naik
hingga 0,71–0,72 sementara AUC val tertahan di ±0,58 (selisih 0,06–0,12). *Loss* val sangat berfluktuasi karena val
hanya memuat 108 positif, sehingga AUC val lebih stabil sebagai kriteria pemilihan model.

![Gambar 17. Kurva loss dan AUC train/val CNN log-mel (3 seed).](../../figures/05_kurva_training_logmel.png)

**Gambar 17.** Kurva *loss* dan AUC train/val CNN *log-mel* (3 *seed*).

**Tabel 13.** Ringkasan CNN pada test Eksperimen A (3 *seed*: 42, 7, 2024).

| Fitur input | AUC val (mean ± std) | AUC test (mean ± std) | AUC test per *seed* (42 / 7 / 2024) | PR-AUC (mean) | AUC test *ensemble* 3 *seed* | Lama latih (menit/*run*) |
|---|---|---|---|---|---|---|
| *Log-mel* (64 × 500) | 0,565 ± 0,004 | 0,572 ± 0,006 | 0,567 / 0,579 / 0,570 | 0,082 | 0,576 | 8,1–20,3 |
| MFCC 2D (40 × 500) | 0,586 ± 0,007 | 0,579 ± 0,030 | 0,544 / 0,591 / 0,601 | 0,088 | 0,591 | 9,2–14,1 |

<!-- sumber: reports/tables/05_cnn_ringkasan.csv, 05_cnn_per_seed.csv, 05_overfitting.csv -->

Kedua CNN berada pada rentang AUC yang sama dengan ML klasik (Tabel 13). Variasi antar-*seed* pada MFCC 2D cukup
besar (0,544–0,601), dan kebetulan *seed* 42 yang ditetapkan sebagai baris utama memberi nilai terendah. Hasil
*ensemble* tiga *seed* (0,576 dan 0,591) hanya dilaporkan sebagai informasi, bukan sebagai model terpilih, karena
pemilihannya tidak ditetapkan sebelumnya.

Grad-CAM pada CNN *log-mel* (Gambar 18) menunjukkan bahwa perhatian model terpusat pada segmen batuk, pada pita
±0,2–1 kHz (puncak ±0,5 kHz) dan > 5,5 kHz. Profil frekuensinya **hampir identik** untuk kedua kelas. Massa Grad-CAM
yang jatuh pada area *padding* hanya ±4%, padahal area *padding* mencakup ±57% luas input. Dengan demikian, CNN
tidak memanfaatkan *padding* atau durasi sebagai jalan pintas.

![Gambar 18. Peta Grad-CAM CNN log-mel (seed 42) pada rekaman test.](../../figures/05_gradcam.png)

**Gambar 18.** Peta Grad-CAM CNN *log-mel* (*seed* 42) pada rekaman test: per kelas, satu contoh benar paling yakin
dan satu contoh acak.

## 4.6 Perbandingan Final dan Uji Signifikansi

Gambar 19 menggabungkan kurva ROC/PR ML terbaik (SVM), kedua CNN, dan *baseline dummy*. Semua kurva model berhimpit
dan hanya sedikit di atas diagonal acak. Uji DeLong (Tabel 14) menegaskan hal tersebut.

- **Tidak ada pasangan model di A yang berbeda bermakna.** Selisih k-NN − SVM adalah +0,004 (p = 0,87), SVM − CNN
  *log-mel* +0,009 (p = 0,70), dan SVM − CNN MFCC 2D +0,032 (p = 0,39).
- **Bahkan SVM vs *dummy* tidak bermakna di A** (Δ = +0,060, p = 0,12), meskipun batas bawah CI AUC SVM (0,523)
  berada di atas 0,5.
- Di B, SVM jauh di atas *dummy* (Δ = +0,339, p < 0,001), tetapi tidak berbeda dari RF (p = 0,62).

**Tabel 14.** Uji signifikansi selisih ROC-AUC pada test yang sama (DeLong; *bootstrap* berpasangan 2.000×).

| Eksp. | Perbandingan | AUC 1 | AUC 2 | Selisih [CI 95% DeLong] | p DeLong | p *bootstrap* |
|---|---|---|---|---|---|---|
| A | k-NN vs SVM (RBF) | 0,580 | 0,576 | +0,004 [−0,040; +0,047] | 0,867 | 0,857 |
| A | SVM (RBF) vs CNN (*log-mel*) | 0,576 | 0,567 | +0,009 [−0,037; +0,055] | 0,696 | 0,719 |
| A | SVM (RBF) vs CNN (MFCC 2D) | 0,576 | 0,544 | +0,032 [−0,040; +0,104] | 0,388 | 0,402 |
| A | SVM (RBF) vs *Dummy* | 0,576 | 0,516 | +0,060 [−0,015; +0,136] | 0,117 | 0,118 |
| B | SVM (RBF) vs *Random Forest* | 0,826 | 0,817 | +0,009 [−0,026; +0,044] | 0,616 | 0,625 |
| B | SVM (RBF) vs *Dummy* | 0,826 | 0,487 | +0,339 [+0,229; +0,450] | < 0,001 | < 0,001 |

<!-- sumber: reports/tables/06_uji_signifikansi_auc.csv -->

![Gambar 19. Kurva ROC dan Precision–Recall gabungan Eksperimen A (test).](../../figures/06_roc_gabungan_A.png)

**Gambar 19.** Kurva ROC dan *Precision–Recall* gabungan Eksperimen A (test): ML terbaik (SVM RBF), CNN *log-mel*,
CNN MFCC 2D, dan *baseline dummy*.

Kriteria seleksi yang ditetapkan sebelumnya (AUC val tertinggi) memilih CNN MFCC 2D *seed* 42 (★). Ironisnya, model
ini justru memiliki AUC test terendah (0,544). Sebaliknya, model dengan AUC test tertinggi (k-NN ▲) termasuk yang
rendah di val. Ketidakstabilan peringkat ini merupakan konsekuensi langsung dari sinyal yang lemah: perbedaan antar-
model jauh lebih kecil daripada ketidakpastian estimasinya.

## 4.7 Eksperimen Keandalan Label

### 4.7.1 Kesepakatan label *self-reported* dan label ahli

Pada 1.011 rekaman Eksperimen B, terdapat 572 rekaman yang memiliki status *self-reported* biner (COVID-19 atau
*healthy*). Tabel 15 menunjukkan bahwa kesepakatan kedua sumber label hanya **50,3%** dengan **Cohen's κ = 0,00**,
yang setara dengan kesepakatan kebetulan. Dari 272 rekaman yang **mengaku** COVID-19, hanya 117 (43%) yang dinilai
COVID-19 oleh pulmonolog. Pola serupa tampak pada metadata lengkap: dari 810 anotasi pada rekaman *self-reported*
COVID-19, hanya 161 yang didiagnosis COVID-19 oleh ahli.

**Tabel 15.** Tabulasi silang status *self-reported* vs diagnosis ahli (*majority vote*) pada rekaman Eksperimen B.

| Status *self-reported* \ diagnosis ahli | COVID-19 | *healthy* | Total |
|---|---|---|---|
| COVID-19 | 117 | 155 | 272 |
| *healthy* | 129 | 171 | 300 |
| *symptomatic* | 173 | 155 | 328 |
| tidak diisi | 51 | 60 | 111 |
| **Total** | **470** | **541** | **1.011** |

<!-- sumber: reports/tables/06_kesepakatan_label_B.csv -->

### 4.7.2 Uji silang sumber label

Uji silang dilakukan dengan membuang dari set evaluasi semua `uuid` yang pernah dipakai pada train/val model sumber.
Hasilnya disajikan pada Tabel 16 dan Gambar 20.

- **S1 (test B, label ahli; n = 102):** model yang dilatih dengan label ahli (B) mencapai AUC 0,84–0,90. Model yang
  dilatih dengan label *self-reported* (A) tetap mencapai 0,61–0,71 (SVM 0,673; RF 0,711; CNN *log-mel* 0,653).
  Artinya, model A ternyata menangkap sebagian pola akustik yang juga dinilai pulmonolog. Pengecualiannya adalah CNN
  MFCC 2D (0,489).
- **S2 (test A, label *self-reported*; n = 1.554):** model B hanya mencapai 0,55–0,58, setara atau sedikit di bawah
  model A (0,57–0,59). Label *self-reported* tidak dapat diprediksi dengan baik, bahkan oleh model yang dilatih
  dengan label yang lebih konsisten.
- **S3 (prediksi identik, dua label; n = 251):** skor model A yang sama menghasilkan AUC 0,52–0,65 terhadap label
  ahli, tetapi hanya 0,49–0,55 terhadap label *self-reported*.

**Tabel 16.** ROC-AUC [CI 95%] uji silang sumber label (set evaluasi ← sumber label pelatihan).

| Model | S1 ← B (ahli) | S1 ← A (*self-rep.*) | S2 ← B (ahli) | S2 ← A (*self-rep.*) | S3 label ahli ← A | S3 label *self-rep.* ← A |
|---|---|---|---|---|---|---|
| *Logistic Regression* | 0,874 [0,797–0,946] | 0,682 [0,576–0,784] | 0,559 [0,500–0,619] | 0,580 [0,515–0,645] | 0,596 [0,528–0,669] | 0,509 [0,391–0,616] |
| SVM (RBF) | 0,896 [0,825–0,960] | 0,673 [0,569–0,779] | 0,545 [0,487–0,604] | 0,585 [0,523–0,647] | 0,574 [0,504–0,648] | 0,541 [0,414–0,646] |
| k-NN | 0,840 [0,753–0,917] | 0,611 [0,498–0,721] | 0,574 [0,513–0,638] | 0,576 [0,516–0,638] | 0,517 [0,442–0,595] | 0,551 [0,437–0,660] |
| *Random Forest* | 0,887 [0,810–0,952] | 0,711 [0,607–0,815] | 0,561 [0,500–0,623] | 0,583 [0,521–0,648] | 0,616 [0,550–0,682] | 0,494 [0,375–0,593] |
| XGBoost | 0,874 [0,796–0,944] | 0,672 [0,569–0,768] | 0,578 [0,518–0,641] | 0,571 [0,505–0,639] | 0,590 [0,524–0,660] | 0,531 [0,420–0,640] |
| CNN (*log-mel*) | – | 0,653 [0,541–0,763] | – | – | 0,652 [0,588–0,719] | 0,489 [0,376–0,600] |
| CNN (MFCC 2D) | – | 0,489 [0,377–0,612] | – | – | 0,518 [0,450–0,590] | 0,516 [0,403–0,635] |

<!-- sumber: reports/tables/06_uji_silang_label.csv. CNN hanya dilatih pada A, sehingga kolom ← B kosong. -->

![Gambar 20. Uji silang sumber label (SVM RBF) pada set evaluasi S1, S2, dan S3.](../../figures/06_uji_silang_label.png)

**Gambar 20.** Uji silang sumber label (SVM RBF): AUC model berlabel *self-reported* (A) vs berlabel ahli (B) pada
set evaluasi berlabel ahli (S1), *self-reported* (S2), dan rekaman yang sama dengan dua label (S3).

Hasil ini harus ditafsirkan dengan hati-hati. Pulmonolog mendiagnosis dengan **mendengarkan rekaman yang sama**,
sehingga label ahli secara konstruksi "konsisten secara akustik". Label ahli bukan *ground truth* virologis seperti
PCR. AUC tinggi pada B karena itu mengukur kesesuaian model dengan penilaian akustik pulmonolog, bukan akurasi
diagnosis infeksi SARS-CoV-2.

## 4.8 Analisis Kesalahan dan Subgrup

### 4.8.1 Distribusi skor dan karakteristik kesalahan

Distribusi skor P(COVID-19) kedua kelas hampir berimpit (Gambar 21). Pada SVM, median skor *healthy* adalah 0,242
dan median skor COVID-19 0,254. Pada CNN *log-mel*, 90% skor test berada di rentang sempit 0,34–0,52. Karakteristik
rekaman per jenis keputusan SVM disajikan pada Tabel 17.

**Tabel 17.** Karakteristik rekaman per jenis keputusan SVM (RBF), test A, ambang 0,249 (nilai median atau persen).

| Jenis | n | Skor median | Durasi *trim* median (s) | Tanpa segmen (%) | Kondisi napas "ya" (%) | Demam "ya" (%) | Laki-laki (%) |
|---|---|---|---|---|---|---|---|
| TP | 60 | 0,283 | 4,72 | 8,3 | 41,7 | 31,7 | 50,0 |
| FN | 48 | 0,219 | 3,30 | 18,8 | 35,4 | 33,3 | 52,1 |
| FP | 677 | 0,278 | 5,00 | 5,5 | 12,7 | 8,7 | 69,3 |
| TN | 839 | 0,215 | 3,05 | 16,8 | 10,8 | 6,9 | 65,9 |

<!-- sumber: reports/tables/06_karakteristik_error_A.csv (proporsi dikali 100) -->

![Gambar 21. Distribusi skor prediksi P(COVID-19) per kelas pada test A; garis putus-putus = ambang Youden dari val.](../../figures/06_distribusi_skor_A.png)

**Gambar 21.** Distribusi skor prediksi P(COVID-19) per kelas pada test A untuk SVM, CNN *log-mel*, dan CNN MFCC 2D;
garis putus-putus = ambang Youden dari val.

Tabel 17 dan Gambar 22 memperlihatkan pola yang konsisten. Rekaman yang diprediksi positif (TP dan FP) cenderung
**lebih panjang dan padat batuk** (median 4,7–5,0 s) dibanding yang diprediksi negatif (3,0–3,3 s). Rekaman FN banyak
yang sangat pendek (0,8–1,0 s pada tiga contoh paling yakin) atau tanpa segmen batuk terdeteksi (18,8%). Skor model
juga berkorelasi dengan durasi: Spearman ρ = 0,41 untuk SVM dan 0,61 untuk CNN *log-mel*, sedangkan CNN MFCC 2D
hampir tidak berkorelasi (−0,05). Padahal durasi saja hanya memberi AUC 0,518. Artinya, sebagian respons model
mencerminkan **kepadatan/panjang episode batuk**, bukan pola akustik yang spesifik untuk COVID-19.

![Gambar 22. Log-mel spectrogram tiga false positive (atas) dan tiga false negative (bawah) paling yakin dari SVM (RBF), Eksperimen A.](../../figures/06_contoh_fp_fn_A.png)

**Gambar 22.** *Log-mel spectrogram* tiga *false positive* (atas) dan tiga *false negative* (bawah) paling yakin dari
SVM (RBF), Eksperimen A.

### 4.8.2 Kinerja per subgrup

Kinerja per subgrup disajikan pada Tabel 18 dan Gambar 23. Temuan yang paling mencolok berkaitan dengan gejala.
Pada subgrup **bergejala**, AUC SVM turun: demam/nyeri otot "ya" 0,507 vs "tidak" 0,590, dan kondisi pernapasan
"ya" 0,529 vs "tidak" 0,586. Prevalensi COVID-19 pada subgrup demam (0,23) juga jauh lebih tinggi daripada pada
subgrup tanpa demam (0,05). Kedua fakta ini menunjukkan bahwa gejala ber-*confound* dengan label. Ketika semua
rekaman sama-sama bergejala, model tidak mampu membedakan batuk COVID-19 dari batuk karena sebab lain. Kinerja
juga rendah pada kelompok usia ≥ 55 tahun (0,475), sedangkan kinerja antar-gender relatif setara (*female* 0,585,
*male* 0,565). Semua CI subgrup lebar karena tiap subgrup hanya memiliki 12–73 positif, sehingga perbedaan
antar-subgrup bersifat indikatif dan belum konklusif.

**Tabel 18.** Kinerja per subgrup pada test A (ambang Youden dari val). AUC dihitung bila subgrup memiliki ≥ 5 positif
dan ≥ 5 negatif.

| Variabel | Subgrup | n | Positif | Prevalensi | AUC SVM [CI 95%] | Sens. SVM | Spes. SVM | AUC CNN *log-mel* |
|---|---|---|---|---|---|---|---|---|
| Demam / nyeri otot | tidak | 1.472 | 73 | 0,050 | 0,590 [0,526–0,660] | 0,562 | 0,558 | 0,563 |
| Demam / nyeri otot | ya | 152 | 35 | 0,230 | 0,507 [0,395–0,613] | 0,543 | 0,496 | 0,532 |
| Kondisi pernapasan | tidak | 1.405 | 66 | 0,047 | 0,586 [0,515–0,656] | 0,530 | 0,559 | 0,546 |
| Kondisi pernapasan | ya | 219 | 42 | 0,192 | 0,529 [0,439–0,616] | 0,595 | 0,514 | 0,585 |
| Gender | *female* | 537 | 50 | 0,093 | 0,585 [0,502–0,670] | 0,540 | 0,579 | 0,568 |
| Gender | *male* | 1.077 | 55 | 0,051 | 0,565 [0,480–0,644] | 0,545 | 0,541 | 0,580 |
| Kelompok umur | < 25 | 289 | 18 | 0,062 | 0,565 [0,416–0,723] | 0,556 | 0,509 | 0,599 |
| Kelompok umur | 25–34 | 424 | 35 | 0,083 | 0,590 [0,479–0,694] | 0,629 | 0,542 | 0,562 |
| Kelompok umur | 35–44 | 388 | 17 | 0,044 | 0,631 [0,509–0,773] | 0,588 | 0,582 | 0,658 |
| Kelompok umur | 45–54 | 241 | 15 | 0,062 | 0,584 [0,430–0,733] | 0,533 | 0,540 | 0,553 |
| Kelompok umur | ≥ 55 | 195 | 12 | 0,062 | 0,475 [0,315–0,625] | 0,333 | 0,601 | 0,556 |

<!-- sumber: reports/tables/06_subgrup_fairness.csv (baris "other" dan "tidak diisi" ada di CSV) -->

![Gambar 23. ROC-AUC [CI 95%] SVM (RBF) per subgrup demografi dan gejala, Eksperimen A (test).](../../figures/06_subgrup_svm_A.png)

**Gambar 23.** ROC-AUC [CI 95%] SVM (RBF) per subgrup demografi dan gejala, Eksperimen A (test).

## 4.9 Perbandingan dengan Literatur

Tabel 19 membandingkan *setting* dan hasil penelitian ini dengan referensi utama.

**Tabel 19.** Perbandingan dengan penelitian terkait.

| Aspek | R2 [@lella2022dcnn] | R3 [@kranthikumar2022lwcnn] | R4 [@xiong2022reliability] | Penelitian ini |
|---|---|---|---|---|
| Dataset | Cambridge COVID-19 Sounds (batuk, napas, suara) | Cambridge COVID-19 Sounds | Batuk *call-centre*, 698 rekaman | COUGHVID v3, batuk saja |
| Label | Dari aplikasi (*self-reported*) | Tidak dinyatakan | PCR vs *self-reported* | A: *self-reported*; B: diagnosis ahli |
| Fitur & model | DAE + GFCC + IMFCC → DCNN | M-MFCC / E-GFCC → CNN ringan $1\times12$ | openSMILE 6.373 fitur → RF dll. | 315 fitur → LR/SVM/k-NN/RF/XGBoost; *log-mel* / MFCC 2D → LightCNN |
| Protokol | k-fold, tidak per subjek; augmentasi tanpa urutan *split* yang jelas | 5 folder, tidak per subjek | Per subjek, uji 63 subjek berlabel PCR | Per `uuid`, *split* sebelum augmentasi/*resampling*; test sekali, prevalensi alami |
| Hasil | Akurasi 95,45% (multimodal); 93,12% (batuk) | Akurasi 91,32% (M-MFCC) | RF AUC 0,83 (PCR) vs 0,79 (*self-reported*) | A: AUC SVM 0,576 (akurasi 0,554); B: AUC 0,826 |

<!-- sumber: reports/tables/06_perbandingan_literatur.csv (nomor halaman sumber di CSV) -->

Angka-angka tersebut **tidak dapat dibandingkan secara langsung** karena dataset, modalitas, label, dan metriknya
berbeda. R2 dan R3 melaporkan akurasi pada data yang relatif seimbang tanpa AUC. Pada test A penelitian ini, akurasi
justru menyesatkan: model *dummy* yang selalu menebak "*healthy*" akan mencapai ±93%. Hasil penelitian ini paling
sejalan dengan R4. Model yang dilatih dan dievaluasi dengan label *self-reported* berkinerja rendah, sedangkan label
yang lebih andal menghasilkan kinerja jauh lebih tinggi dengan fitur yang sama. Arsitektur CNN ringan ala R3
(*kernel* $1\times12$) telah diimplementasikan tetapi belum dilatih (lihat saran).

## 4.10 Pembahasan

**Mengapa kinerja Eksperimen A rendah?** Terdapat empat faktor yang saling memperkuat.

1. **Derau label.** Label *self-reported* tidak terverifikasi. Rekaman bisa dibuat sebelum, saat, atau jauh setelah
   infeksi, dan sebagian peserta mungkin keliru atau tidak jujur. Kesepakatan dengan label ahli setara kebetulan
   (κ = 0,00), dan model B tidak mampu memprediksi label A (S2 ≈ 0,55).
2. **Sinyal akustik yang lemah dan tidak spesifik.** Batuk adalah bunyi nonperiodik yang sangat dipengaruhi
   individu, cara batuk, dan perangkat. Fitur paling diskriminatif hanya mencapai AUC univariat 0,58, dan proyeksi
   PCA/t-SNE tidak menunjukkan pemisahan kelas sama sekali.
3. ***Confounding* oleh gejala dan durasi.** Sinyal yang tersisa sebagian berasal dari karakteristik "batuk
   bergejala" atau "rekaman padat batuk", bukan dari ciri spesifik COVID-19. Hal ini tampak dari turunnya AUC pada
   subgrup bergejala dan korelasi skor dengan durasi.
4. **Variasi perangkat dan lingkungan** pada data *crowdsourced*, ditambah tidak adanya ID subjek.

**Mengapa CNN tidak lebih baik?** Ketika batas atas kinerja ditentukan oleh kualitas label, kapasitas model yang
lebih besar tidak membantu. Model yang lebih fleksibel justru lebih mudah mempelajari derau, seperti terlihat pada
*overfitting* CNN MFCC 2D. Selain itu, jumlah data positif pada train hanya ±500 rekaman, yang tergolong kecil untuk
*deep learning* dari awal, dan *log-mel* yang dipelajari CNN pada dasarnya memuat informasi yang sama dengan MFCC
tabular. Pada data kecil yang berderau, fitur rekayasa (*hand-crafted*) + SVM sama kompetitifnya dengan CNN dan jauh
lebih murah: SVM dilatih dalam hitungan menit, sedangkan CNN membutuhkan ±12 menit per *run*.

**Mengapa label ahli memberi kinerja jauh lebih tinggi?** Pipeline yang sama mencapai AUC 0,83 pada label ahli,
sehingga fitur MFCC/spektral **mampu** menangkap pola akustik yang konsisten. Namun, pola yang konsisten itu adalah
pola yang **didengar pulmonolog**, misalnya batuk yang lebih panjang, lebih "basah", atau lebih bergejala. Pola ini
belum tentu sama dengan status infeksi. Peran durasi yang cukup kuat di B (AUC 0,67 untuk durasi saja) mendukung
tafsiran tersebut.

**Implikasi praktis.** Dengan AUC ±0,58 dan *precision* ±0,08 pada prevalensi 6,7%, model berbasis label
*self-reported* **tidak layak** dipakai untuk skrining COVID-19. Pada ambang yang dipilih, model meloloskan hampir
separuh kasus positif dan sekaligus menandai ±45% orang sehat sebagai positif. Hasil ini menekankan tiga hal
berikut.

1. Kualitas label, idealnya terkonfirmasi PCR dan dengan ID subjek, lebih penting daripada pemilihan arsitektur
   model.
2. Evaluasi harus memakai prevalensi realistis, metrik ROC-AUC/PR-AUC dengan interval kepercayaan, dan uji
   signifikansi. Akurasi pada data seimbang tidak memadai.
3. Klaim kinerja > 90% di literatur perlu dibaca bersama protokolnya, terutama urutan augmentasi terhadap *split*,
   pembagian per subjek, dan metrik yang dipakai.
