# BAB I PENDAHULUAN

## 1.1 Latar Belakang

Pandemi COVID-19 menuntut kegiatan pengujian massal sebagai sarana utama untuk memantau dan menekan penyebaran
penyakit. Baku emas (*gold standard*) diagnosis SARS-CoV-2 adalah *reverse transcription polymerase chain reaction*
(RT-PCR). Pemeriksaan ini bersifat invasif karena memerlukan usap tenggorok/nasofaring, membutuhkan fasilitas
laboratorium, dan hasilnya dapat tertunda beberapa hari ketika kapasitas pengujian terbatas [@xiong2022reliability].
Kondisi tersebut mendorong pencarian metode skrining yang murah, cepat, non-invasif, dan dapat dilakukan dari jarak
jauh. Salah satu kandidatnya adalah analisis suara pernapasan, khususnya suara batuk, yang dapat direkam dengan
mikrofon telepon pintar atau peramban web [@lella2021review].

Batuk merupakan gejala umum berbagai infeksi saluran pernapasan, termasuk COVID-19. Infeksi pada saluran napas
dapat mengubah karakteristik akustik batuk, misalnya durasi fase eksplosif, kandungan energi pada pita frekuensi
tertentu, dan "warna" suara (*timbre*). Dalam kerangka pengenalan pola suara, perubahan tersebut dapat ditangkap
melalui representasi sinyal di domain waktu maupun domain frekuensi, kemudian dipelajari oleh algoritma
klasifikasi. Selama pandemi, sejumlah basis data suara batuk dikumpulkan secara *crowdsourcing*, antara lain
Cambridge *COVID-19 Sounds* [@brown2020sounds], Coswara [@sharma2020coswara], dan COUGHVID [@orlandic2021coughvid].
Beberapa penelitian melaporkan kinerja yang sangat tinggi. Laguarta dkk. [@laguarta2020] melaporkan akurasi 97,1% dengan
MFCC dan ResNet50 (sebagaimana dirangkum [@lella2021review]), sedangkan Lella dan Pja [@lella2022dcnn] serta Kranthi Kumar dan Alphonse [@kranthikumar2022lwcnn] melaporkan
akurasi 91–95% dengan fitur MFCC/GFCC dan CNN.

Namun, label pada dataset *crowdsourced* umumnya bersifat *self-reported*: peserta sendiri yang menyatakan apakah
mereka positif COVID-19, sehat, atau bergejala, tanpa verifikasi PCR. Xiong dkk. [@xiong2022reliability]
menunjukkan bahwa model yang dilatih dengan label *self-reported* berkinerja lebih rendah dan memanfaatkan fitur
yang kurang stabil dibanding model yang dilatih dengan label terkonfirmasi PCR. Selain itu, sebagian angka yang
sangat tinggi di literatur diperoleh dengan protokol yang berisiko kebocoran data (*data leakage*), misalnya
augmentasi sebelum pembagian data atau pembagian yang tidak memperhatikan subjek, serta metrik akurasi pada data
yang telah diseimbangkan. Karena itu, perlu ada evaluasi yang jujur dan dapat direproduksi pada dataset terbuka
berskala besar dengan prevalensi kelas yang realistis.

Penelitian ini merupakan studi kasus pengenalan pola suara sesuai materi Pertemuan 8 mata kuliah Pengenalan Pola
dan Kecerdasan Buatan [@sela2026materi]. Penelitian mengikuti alur akuisisi → *preprocessing* → ekstraksi fitur →
klasifikasi → evaluasi, dan memakai dataset COUGHVID v3 [@orlandic2021coughvid]. Dua keluarga metode dibandingkan:
(1) *machine learning* (ML) klasik pada fitur tabular (energi, *zero-crossing rate*, fitur spektral, *pitch*, MFCC,
dan GFCC), serta (2) *convolutional neural network* (CNN) ringan pada representasi dua dimensi (*log-mel
spectrogram* dan MFCC 2D). Karena keandalan label menjadi isu utama, penelitian ini juga membandingkan label
*self-reported* dengan label diagnosis pulmonolog yang tersedia untuk sebagian rekaman COUGHVID.

## 1.2 Rumusan Masalah

1. Bagaimana merancang *pipeline* pengenalan pola suara batuk, mulai dari akuisisi, *preprocessing*, dan ekstraksi
   fitur hingga klasifikasi, yang bebas kebocoran data pada dataset COUGHVID v3?
2. Seberapa baik fitur akustik MFCC, GFCC, fitur spektral, dan fitur domain waktu membedakan suara batuk COVID-19
   dari batuk orang sehat, dan kelompok fitur mana yang paling informatif?
3. Bagaimana perbandingan kinerja model ML klasik (*Logistic Regression*, SVM, k-NN, *Random Forest*, XGBoost)
   dengan CNN ringan pada *log-mel spectrogram* dan MFCC 2D, diukur dengan ROC-AUC, PR-AUC, sensitivitas, dan
   spesifisitas pada data uji berprevalensi alami?
4. Sejauh mana keandalan label (*self-reported* vs diagnosis ahli) memengaruhi kinerja klasifikasi?

## 1.3 Tujuan Penelitian

1. Membangun *pipeline* lengkap dan reprodusibel (seed 42) untuk klasifikasi suara batuk COVID-19 vs sehat pada
   COUGHVID v3, dengan pembagian data dilakukan sebelum *oversampling*, augmentasi, *scaling*, dan seleksi fitur.
2. Mengekstraksi serta menganalisis fitur akustik domain waktu dan domain frekuensi, lalu mengukur kontribusi tiap
   kelompok fitur melalui uji statistik dan ablasi.
3. Membandingkan model ML klasik dan CNN ringan secara statistik (interval kepercayaan *bootstrap* dan uji DeLong).
4. Mengukur pengaruh sumber label melalui eksperimen berlabel ahli dan uji silang label.

## 1.4 Manfaat Penelitian

1. **Akademik:** menjadi contoh penerapan konsep pengenalan pola suara, seperti *sampling*, kuantisasi, FFT,
   *spectrogram*, MFCC, dan GFCC, pada kasus nyata yang lengkap dari data mentah hingga evaluasi.
2. **Metodologis:** menyediakan *baseline* jujur pada COUGHVID dengan protokol anti-kebocoran, prevalensi uji alami,
   dan metrik yang sesuai untuk data tidak seimbang. *Baseline* ini dapat menjadi pembanding bagi klaim kinerja
   tinggi di literatur.
3. **Praktis:** memberi gambaran realistis tentang kelayakan skrining COVID-19 berbasis batuk serta faktor yang
   membatasinya, terutama kualitas label, sebagai masukan bagi pengembangan sistem skrining berikutnya.

## 1.5 Batasan Masalah

1. Data yang dipakai hanya COUGHVID v3 (rekaman hingga Oktober 2021) dan hanya modalitas batuk, tanpa suara napas
   atau suara ucapan.
2. Rekaman disaring dengan `cough_detected` ≥ 0,8 sesuai rekomendasi penyusun dataset [@orlandic2021coughvid].
3. Klasifikasi bersifat biner. Eksperimen A membandingkan COVID-19 dengan *healthy* berdasarkan label
   *self-reported*. Eksperimen B membandingkan COVID-19 dengan *healthy cough* berdasarkan diagnosis pulmonolog.
   Kelas *symptomatic* tidak dipakai.
4. Tidak tersedia label PCR dan ID subjek, sehingga pembagian data dilakukan per rekaman (`uuid`).
5. Komputasi hanya memakai CPU tanpa GPU, sehingga model *deep learning* dibatasi pada CNN ringan (±97 ribu
   parameter) dengan input yang diturunkan resolusinya.
6. Sistem yang dibangun adalah prototipe penelitian, bukan alat diagnosis klinis.

## 1.6 Sistematika Penulisan

Bab I memuat latar belakang, rumusan masalah, tujuan, manfaat, dan batasan. Bab II membahas dasar teori suara
digital, ekstraksi fitur, algoritma klasifikasi, dan penelitian terkait. Bab III menjelaskan metodologi, mencakup
dataset, *preprocessing*, fitur, model, dan protokol evaluasi. Bab IV menyajikan hasil dan pembahasan. Bab V berisi
kesimpulan dan saran.
