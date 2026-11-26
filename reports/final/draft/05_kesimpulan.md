# BAB V KESIMPULAN DAN SARAN

## 5.1 Kesimpulan

Kesimpulan berikut menjawab keempat rumusan masalah pada Subbab 1.2.

1. **Pipeline pengenalan pola suara (rumusan masalah 1).** Penelitian ini berhasil membangun *pipeline* lengkap
   dan reprodusibel sesuai alur akuisisi → *preprocessing* → ekstraksi fitur → klasifikasi → evaluasi pada
   COUGHVID v3. Rinciannya sebagai berikut.
   - Rekaman *crowdsourced* yang formatnya beragam (webm/ogg/wav, 16–48 kHz) didekode seragam menjadi WAV 16 kHz
     mono PCM 16-bit, lalu di-*trim*, disegmentasi, direduksi derau, dan dinormalisasi. Dari 5.881 rekaman yang
     diproses, hanya 17 yang gugur.
   - Kebocoran data dicegah dengan membagi data per `uuid` terlebih dahulu. *Undersampling*, SMOTE, augmentasi,
     *scaling*, dan seleksi fitur hanya di-*fit* pada train. Ambang keputusan dipilih di val, dan test dievaluasi
     satu kali pada prevalensi alami (6,7%).
2. **Daya beda fitur (rumusan masalah 2).** Dengan label *self-reported*, fitur akustik hanya memiliki daya beda
   yang sangat lemah.
   - Hanya 39 dari 315 fitur yang berbeda signifikan, dan semuanya berefek kecil. Fitur terkuat adalah kontras
     spektral pita tertinggi (AUC univariat 0,58).
   - Pada ablasi, MFCC (+Δ, ΔΔ) adalah kelompok fitur tunggal terbaik (AUC 0,568 pada A; 0,813 pada B). GFCC
     redundan terhadap MFCC, dan fitur domain waktu (RMS, ZCR) paling lemah.
   - Durasi tidak menjadi jalan pintas bagi model penuh, tetapi merupakan *confounder* potensial pada label ahli.
3. **ML klasik vs CNN ringan (rumusan masalah 3).** Pada Eksperimen A, kedua keluarga model berkinerja setara dan
   hanya sedikit di atas acak.
   - SVM RBF (ML terbaik menurut val) mencapai ROC-AUC 0,576 [0,523–0,629] dan PR-AUC 0,088.
   - CNN *log-mel* mencapai 0,567 (3 *seed*: 0,572 ± 0,006), sedangkan CNN MFCC 2D mencapai 0,544
     (3 *seed*: 0,579 ± 0,030).
   - Uji DeLong tidak menemukan perbedaan bermakna antar-model (p = 0,39–0,87), bahkan antara SVM dan *dummy*
     (p = 0,12).
   - CNN ringan tidak memberi keunggulan atas ML klasik, padahal biaya komputasinya lebih besar.
   - Dengan *precision* ±0,08, model **tidak layak** dipakai untuk skrining.
4. **Pengaruh keandalan label (rumusan masalah 4).** Keandalan label adalah faktor pembatas utama.
   - Dengan fitur dan *pipeline* yang identik, label diagnosis pulmonolog menghasilkan ROC-AUC 0,826 [0,745–0,888],
     jauh di atas label *self-reported* (0,576).
   - Kedua sumber label praktis tidak sepakat (κ = 0,00; hanya 43% pengaku COVID-19 dinilai COVID-19 oleh ahli).
   - Uji silang menunjukkan bahwa model berlabel *self-reported* tetap menangkap sebagian pola yang dinilai ahli
     (AUC 0,61–0,71 pada test berlabel ahli). Sebaliknya, label *self-reported* tidak dapat diprediksi bahkan oleh
     model berlabel ahli (AUC 0,55–0,58).
   - Temuan ini sejalan dengan R4 [1]. Namun, label ahli diperoleh dari mendengar rekaman yang
     sama, sehingga AUC tinggi pada label ahli mencerminkan kesesuaian dengan penilaian akustik pulmonolog, bukan
     akurasi diagnosis virologis.

Penelitian ini juga memiliki sejumlah keterbatasan. Label tidak terkonfirmasi PCR, dan tidak tersedia ID subjek
sehingga pembagian dilakukan per rekaman. Rekaman berasal dari perangkat yang beragam, dan demografinya bias
(±66% pria pada test A). *Undersampling* train A membuang 5.585 rekaman *healthy* dan menggeser kalibrasi
probabilitas. Segmentasi batuk tidak sempurna (11,6% rekaman tanpa segmen terdeteksi). Ukuran test B kecil
(152 rekaman), dan validasi silang di train tidak berbentuk *nested CV*.

## 5.2 Saran

1. **Validasi eksternal dengan label PCR.** Model sebaiknya dilatih atau diuji lintas dataset berlabel lebih andal,
   misalnya Coswara [4] atau COVID-19 Sounds [3], dengan pembagian per subjek.
2. **Penanganan label berderau.** Label `status_SSL` (Eksperimen C) dapat dicoba, demikian pula teknik pelatihan
   yang tahan derau label seperti *label smoothing* dan *co-teaching*.
3. **Fitur per episode batuk.** Fitur dapat diagregasi per segmen batuk agar model tidak bergantung pada durasi atau
   jumlah batuk. Representasi *embedding* pra-latih (VGGish, PANNs, wav2vec 2.0) atau set fitur openSMILE ComParE
   seperti pada R4 juga layak dicoba.
4. **Kalibrasi dan ambang klinis.** Probabilitas keluaran perlu dikalibrasi (Platt/isotonik di val), dan ambang
   sebaiknya dipilih berdasarkan target sensitivitas skrining, misalnya ≥ 0,90, disertai laporan spesifisitas yang
   dikorbankan.
5. **Metadata gejala.** Informasi demam dan kondisi pernapasan dapat digabungkan secara eksplisit, kemudian diuji
   apakah suara batuk memberi informasi tambahan di atas gejala (*incremental value*).
6. **Eksplorasi arsitektur.** Augmentasi level *waveform* (*time stretch*, *pitch shift*, derau) seperti pada R2,
   CNN dengan *kernel* $1\times12$ seperti pada R3, CRNN, dan *ensemble* antar-*seed* dapat dicoba, dengan syarat
   pemilihan model tetap ditetapkan sebelum test dilihat.
