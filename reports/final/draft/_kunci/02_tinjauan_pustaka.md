# BAB II TINJAUAN PUSTAKA

## 2.1 Dasar Teori Suara

### 2.1.1 Gelombang suara, amplitudo, frekuensi, dan periode

Suara adalah gelombang mekanik longitudinal, yaitu rapatan dan renggangan tekanan udara yang merambat dari sumber
getar ke telinga atau mikrofon [@sela2026materi]. Tiga besaran dasar gelombang suara adalah sebagai berikut.

- **Amplitudo** ($A$) adalah simpangan maksimum tekanan dari keadaan setimbang. Amplitudo berkaitan dengan
  kenyaringan (*loudness*) dan biasanya dinyatakan dalam skala logaritmik desibel:
  $L = 20\log_{10}(A/A_{\text{ref}})$ dB.
- **Frekuensi** ($f$, Hz) adalah banyaknya getaran per detik. Telinga manusia menangkap ±20 Hz–20 kHz.
- **Periode** ($T$, detik) adalah waktu satu getaran penuh, dengan hubungan $T = 1/f$.

Sebuah nada murni dapat ditulis sebagai $x(t) = A\sin(2\pi f t + \phi)$, dengan $\phi$ sebagai fase awal.

### 2.1.2 *Pitch* dan *timbre*

***Pitch*** adalah persepsi tinggi-rendahnya nada. Secara fisik, *pitch* terutama ditentukan oleh frekuensi dasar
(*fundamental frequency*, $f_0$) dari suara yang periodik. ***Timbre*** ("warna" suara) membedakan dua suara dengan
*pitch* dan kenyaringan yang sama, misalnya suara biola dan piano pada nada yang sama. *Timbre* ditentukan oleh
susunan harmonik, selubung spektrum (*spectral envelope*), dan dinamika temporal (*attack–decay*). Fitur seperti MFCC
dan GFCC pada dasarnya adalah ringkasan numerik dari selubung spektrum, sehingga dapat dipandang sebagai
deskriptor *timbre* [@sela2026materi].

### 2.1.3 Suara periodik dan nonperiodik

Suara **periodik** berulang dengan pola yang sama setiap periode $T_0$, misalnya vokal atau nada alat musik, dan
memiliki spektrum berupa harmonik-harmonik pada kelipatan $f_0$. Suara **nonperiodik** tidak memiliki pola berulang,
misalnya desis, letupan, atau derau, sehingga spektrumnya kontinu dan tersebar lebar. **Batuk didominasi komponen
nonperiodik.** Satu episode batuk umumnya terdiri atas fase eksplosif (letupan pembukaan glotis yang mirip impuls),
fase intermediet (aliran turbulen yang mirip derau), dan kadang fase bersuara (*voiced*) singkat ketika pita suara
ikut bergetar. Karena itu, *pitch* hanya terdefinisi pada sebagian kecil *frame* batuk. Fitur spektral dan cepstral
lebih tepat dipakai untuk merepresentasikan batuk.

### 2.1.4 Sinyal analog dan digital: *sampling* dan kuantisasi

Mikrofon mengubah tekanan udara menjadi sinyal listrik **analog** $x_a(t)$ yang kontinu dalam waktu dan amplitudo.
Untuk diolah komputer, sinyal diubah menjadi sinyal **digital** melalui dua langkah [@oppenheim2010].

1. ***Sampling*** (pencuplikan waktu): $x[n] = x_a(nT_s)$ dengan periode cuplik $T_s = 1/f_s$. Menurut teorema
   Nyquist–Shannon, sinyal dapat direkonstruksi tanpa *aliasing* bila $f_s \ge 2 f_{\max}$. Frekuensi $f_s/2$
   disebut frekuensi Nyquist. Penelitian ini memakai $f_s = 16$ kHz, sehingga informasi hingga 8 kHz tetap
   terjaga. Rentang ini mencakup sebagian besar energi batuk dan ucapan.
2. **Kuantisasi** (pencuplikan amplitudo): setiap cuplikan dibulatkan ke salah satu dari $2^B$ level, dengan $B$
   sebagai *bit depth*. PCM 16-bit memberi $2^{16} = 65.536$ level dengan rasio sinyal terhadap derau kuantisasi
   teoretis $\text{SQNR} \approx 6{,}02B + 1{,}76 \approx 98$ dB.

### 2.1.5 *Waveform*, domain waktu, dan domain frekuensi

***Waveform*** adalah grafik amplitudo terhadap waktu, yaitu representasi **domain waktu**. Dari domain waktu
dapat diamati durasi, selubung energi, dan saat terjadinya batuk, tetapi kandungan frekuensinya tidak tampak
langsung. Representasi **domain frekuensi** menunjukkan seberapa besar energi pada tiap frekuensi, dan diperoleh
melalui transformasi Fourier.

### 2.1.6 Transformasi Fourier dan FFT

*Discrete Fourier Transform* (DFT) menguraikan $N$ cuplikan sinyal menjadi $N$ komponen sinusoidal:

$$X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi k n / N}, \qquad k = 0,\dots,N-1,$$

dengan resolusi frekuensi $\Delta f = f_s/N$. *Fast Fourier Transform* (FFT) adalah algoritma untuk menghitung DFT
dengan kompleksitas $O(N\log N)$, jauh lebih cepat daripada $O(N^2)$ [@oppenheim2010]. Dengan $N = 512$ pada
16 kHz, resolusinya $16.000/512 = 31{,}25$ Hz per *bin*.

### 2.1.7 STFT dan *spectrogram*

Sinyal batuk bersifat nonstasioner karena isi frekuensinya berubah cepat. Karena itu, FFT diterapkan pada potongan
pendek (*frame*) yang dikalikan fungsi jendela $w[n]$ dan digeser sejauh *hop* $H$. Proses ini disebut
*Short-Time Fourier Transform* (STFT):

$$X[m,k] = \sum_{n=0}^{N-1} x[n + mH]\, w[n]\, e^{-j 2\pi k n / N}.$$

***Spectrogram*** adalah $|X[m,k]|^2$ (atau dalam dB) yang ditampilkan sebagai citra waktu × frekuensi. Ukuran
*frame* menentukan kompromi resolusi: *frame* panjang memberi resolusi frekuensi baik tetapi resolusi waktu buruk,
dan sebaliknya. Penelitian ini memakai *frame* 32 ms (512 cuplikan), *hop* 10 ms (160 cuplikan), dan jendela Hann.

## 2.2 Ekstraksi Fitur Suara

### 2.2.1 Fitur domain waktu: energi dan ZCR

- **Energi / RMS** per *frame*: $E_m = \sum_{n} x_m[n]^2$, $\text{RMS}_m = \sqrt{E_m / N}$. Fitur ini
  menggambarkan kenyaringan sesaat; letupan batuk tampak sebagai puncak RMS.
- ***Zero-Crossing Rate*** (ZCR): $\text{ZCR}_m = \frac{1}{2N}\sum_{n}\left|\operatorname{sgn}(x_m[n]) -
  \operatorname{sgn}(x_m[n-1])\right|$. ZCR tinggi menandakan bunyi berfrekuensi tinggi atau mirip derau
  (nonperiodik), sedangkan ZCR rendah menandakan bunyi bersuara (periodik).

### 2.2.2 Fitur spektral

Dari magnitudo STFT $|X[m,k]|$ pada frekuensi $f_k$ dihitung beberapa fitur berikut.

- ***Spectral centroid*** ("pusat massa" spektrum): $C_m = \sum_k f_k |X[m,k]| \,/\, \sum_k |X[m,k]|$.
  Fitur ini berkorelasi dengan persepsi "terang" suatu bunyi.
- ***Spectral bandwidth***: sebaran spektrum di sekitar *centroid*.
- ***Spectral rolloff***: frekuensi di bawah batas yang memuat 85% energi spektrum.
- ***Spectral flatness***: rasio rerata geometrik terhadap rerata aritmetik spektrum. Nilainya mendekati 1 untuk
  derau putih dan mendekati 0 untuk nada murni.
- ***Spectral contrast***: selisih (dB) antara puncak dan lembah spektrum pada beberapa sub-pita oktaf.

### 2.2.3 *Mel-Frequency Cepstral Coefficients* (MFCC)

MFCC [@davis1980] meniru persepsi telinga manusia yang lebih peka terhadap perbedaan frekuensi rendah daripada
frekuensi tinggi. Tahapannya sebagai berikut.

1. ***Pre-emphasis***: $y[n] = x[n] - \alpha x[n-1]$ dengan $\alpha \approx 0{,}97$. Langkah ini memperkuat
   komponen frekuensi tinggi yang teredam.
2. ***Framing***: sinyal dipotong menjadi *frame* pendek (20–40 ms) yang saling tumpang-tindih, sehingga sinyal di
   dalam satu *frame* dapat dianggap kuasi-stasioner.
3. ***Windowing***: setiap *frame* dikalikan fungsi jendela, misalnya Hamming
   $w[n] = 0{,}54 - 0{,}46\cos\!\big(2\pi n/(N-1)\big)$ atau Hann, untuk mengurangi kebocoran spektral di tepi
   *frame*.
4. **FFT**: dihitung spektrum daya $P[k] = |X[k]|^2 / N$.
5. ***Mel filterbank***: spektrum daya dilewatkan pada $M$ filter segitiga yang berjarak sama pada skala mel

   $$\text{mel}(f) = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right),$$

   sehingga filter di frekuensi rendah sempit dan rapat, sedangkan di frekuensi tinggi lebar dan jarang
   [@oshaughnessy1987]. Keluarannya adalah energi tiap pita mel $S_m$.
6. **Logaritma**: $\log S_m$ meniru persepsi kenyaringan yang logaritmik dan mengubah perkalian sumber × filter
   menjadi penjumlahan.
7. ***Discrete Cosine Transform*** (DCT-II): $c_i = \sum_{m=1}^{M} \log S_m \cos\!\left[\frac{\pi i}{M}(m -
   \tfrac{1}{2})\right]$. DCT mendekorelasi energi pita dan memadatkan informasi selubung spektrum ke beberapa
   koefisien pertama. Biasanya 12–13 koefisien pertama dipakai.

Dinamika temporal ditangkap dengan koefisien **delta** (Δ, turunan pertama) dan **delta-delta** (ΔΔ, turunan
kedua) dari MFCC antar-*frame*. Matriks $\log S_m$ sebelum DCT disebut ***log-mel spectrogram***. Representasi ini
lazim dipakai sebagai "citra" input CNN.

### 2.2.4 *Gammatone Frequency Cepstral Coefficients* (GFCC)

GFCC [@shao2009gfcc] mengganti filter segitiga mel dengan **filterbank gammatone** yang memodelkan respons membran
basilar koklea:

$$g(t) = a\, t^{\,n-1} e^{-2\pi b\, \text{ERB}(f_c)\, t} \cos(2\pi f_c t + \phi),$$

dengan orde $n = 4$ dan lebar pita mengikuti skala *Equivalent Rectangular Bandwidth*
$\text{ERB}(f) = 24{,}7\,(4{,}37 f/1000 + 1)$. Energi tiap filter dikompresi dengan akar pangkat tiga, bukan
logaritma, lalu ditransformasi dengan DCT. GFCC dilaporkan lebih tahan derau daripada MFCC dan menjadi salah satu
kanal fitur pada [@lella2022dcnn] dan [@kranthikumar2022lwcnn].

### 2.2.5 Estimasi *pitch*

Frekuensi dasar $f_0$ pada bagian bersuara diestimasi dengan pYIN [@mauch2014pyin]. pYIN adalah versi probabilistik
algoritma YIN yang juga mengeluarkan probabilitas sebuah *frame* bersuara. Rasio *frame* bersuara (*voiced ratio*)
menjadi ukuran sederhana porsi periodik dalam rekaman batuk.

## 2.3 Algoritma Klasifikasi

### 2.3.1 *Support Vector Machine* (SVM)

SVM [@cortes1995] mencari *hyperplane* pemisah dengan margin maksimum. Dengan *kernel trick*, fungsi keputusannya
$f(\mathbf{x}) = \operatorname{sgn}\big(\sum_i \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\big)$ dapat memisahkan
data secara nonlinear. *Kernel* RBF $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \lVert \mathbf{x} - \mathbf{x}'
\rVert^2)$ dikendalikan oleh $\gamma$ (jangkauan pengaruh tiap titik) dan $C$ (penalti kesalahan). SVM cocok untuk
data berdimensi tinggi dengan jumlah sampel menengah, seperti vektor 315 fitur akustik.

### 2.3.2 *Random Forest* (RF)

RF [@breiman2001] adalah *ensemble* banyak pohon keputusan. Setiap pohon dilatih pada sampel *bootstrap* dan, di
setiap simpul, hanya mempertimbangkan subset acak fitur. Prediksi diperoleh dari rata-rata suara pohon. Pengacakan
ini menurunkan varians dan memberi ukuran kepentingan fitur (*mean decrease impurity*).

### 2.3.3 XGBoost

XGBoost [@chen2016xgb] membangun pohon secara berurutan (*gradient boosting*). Setiap pohon baru memperbaiki residu
pohon sebelumnya dengan meminimalkan fungsi objektif ber-regularisasi
$\mathcal{L} = \sum_i \ell(y_i, \hat{y}_i) + \sum_t \big(\gamma T_t + \tfrac{1}{2}\lambda \lVert w_t \rVert^2\big)$,
yang dihampiri dengan ekspansi Taylor orde kedua. Parameter `scale_pos_weight` dapat dipakai untuk menangani kelas
tidak seimbang.

Sebagai pembanding, penelitian ini juga memakai *Logistic Regression* (model linear probabilistik) dan k-NN
(klasifikasi berdasarkan $k$ tetangga terdekat).

### 2.3.4 *Convolutional Neural Network* (CNN)

CNN [@lecun2015] mempelajari fitur secara hierarkis langsung dari data berbentuk *grid*, misalnya citra atau
*spectrogram*. Lapisan konvolusi menerapkan filter kecil yang sama di seluruh posisi, sehingga pola lokal seperti
tepi atau *formant* dapat dikenali di mana pun pola itu muncul pada sumbu waktu. Komponen yang dipakai pada
penelitian ini adalah:

- **konvolusi 2D** $3\times3$ + ***batch normalization*** [@ioffe2015] + ReLU;
- ***max pooling*** $2\times2$ untuk mereduksi resolusi dan memberi invariansi pergeseran kecil;
- ***dropout*** untuk regularisasi;
- ***global average pooling*** (GAP), yang merata-ratakan setiap peta fitur sehingga panjang input boleh bervariasi
  dan jumlah parameter tetap kecil;
- lapisan *fully connected* dengan keluaran sigmoid untuk klasifikasi biner.

Sebagai "citra", *spectrogram* berbeda dari foto karena sumbu vertikalnya adalah frekuensi yang tidak invarian
terhadap translasi. Meskipun demikian, CNN terbukti efektif pada klasifikasi audio. R3 [@kranthikumar2022lwcnn]
mengusulkan CNN ringan dengan *receptive field* $1\times12$ dan lima lapisan untuk suara pernapasan.

## 2.4 Penelitian Terkait

Tabel 1 merangkum lima referensi utama. Angka hasil dikutip dari naskah aslinya. Ketiga penelitian eksperimental
(R2–R4) **tidak memakai COUGHVID**, sehingga angkanya tidak dapat dibandingkan langsung dengan penelitian ini
(lihat Subbab 4.9).

**Tabel 1.** Ringkasan penelitian terkait.

| No. | Penulis (tahun) | Dataset | Metode | Hasil | Keterbatasan |
|---|---|---|---|---|---|
| R0 | Orlandic, Teijeiro & Atienza (2021) [@orlandic2021coughvid] | COUGHVID: rekaman batuk *crowdsourced* via aplikasi web (>20.000 rekaman pada versi awal; v3 berisi 34.434 rekaman) | Pengumpulan data, detektor batuk berbasis XGBoost (skor `cough_detected`), anotasi 4 pulmonolog untuk sebagian rekaman | Korpus batuk terbuka terbesar dengan metadata demografi, gejala, dan label ahli | Label COVID-19 *self-reported*; tanpa ID subjek; kualitas rekaman bervariasi; bukan *benchmark* diagnosis |
| R1 | Lella & Alphonse (2021) [@lella2021review] | Tinjauan dataset *crowdsourced* (COUGHVID, Coswara, COVID-19 Sounds, dll.) | Tinjauan literatur metode MFCC, SVM, CNN, dan RNN untuk suara pernapasan | Merangkum akurasi 80–97% dari berbagai studi (mis. Laguarta dkk. 97,1%) | Tinjauan naratif tanpa evaluasi ulang; angka antar-studi tidak sebanding karena data dan protokol berbeda |
| R2 | Lella & Pja (2022) [@lella2022dcnn] | Cambridge COVID-19 Sounds (batuk, napas, suara) | *De-noising autoencoder* + GFCC + IMFCC sebagai kanal fitur → DCNN *softmax*; augmentasi (*time stretch*, *pitch shift*, derau) | Akurasi 95,45% / F1 96,96% (multimodal); batuk saja akurasi 93,12% | Tidak ada AUC; pembagian tidak per subjek; urutan augmentasi terhadap *split* tidak dinyatakan (risiko kebocoran) |
| R3 | Kranthi Kumar & Alphonse (2022) [@kranthikumar2022lwcnn] | Cambridge COVID-19 Sounds | M-MFCC & E-GFCC → CNN ringan (*receptive field* $1\times12$, 5 lapisan, *batch norm*, 100 *epoch*) | Akurasi 91,32% / F1 92,48% (M-MFCC); multimodal akurasi 92,32% | Hanya akurasi/F1; jumlah data dan pembagian tidak konsisten di naskah; tidak per subjek |
| R4 | Xiong dkk. (2022) [@xiong2022reliability] | Rekaman batuk *call-centre* (698 rekaman, 1 per subjek); uji berlabel PCR | openSMILE 6.373 fitur + seleksi ElasticNet → LR, LDA, RF, XGB, DNN, CNN | RF AUC 0,83 (dilatih label PCR) vs 0,79 (label *self-reported*); label *self-reported* kurang andal | Data kecil dan dari satu sumber; tidak dipublikasikan terbuka; uji hanya 63 subjek |

<!-- sumber: reports/tables/06_perbandingan_literatur.csv; R1 dari PDF references/ -->

Dari Tabel 1 terlihat dua celah. Pertama, studi dengan akurasi tinggi (R2, R3) memakai metrik akurasi/F1, data yang
relatif seimbang, dan protokol yang tidak sepenuhnya mencegah kebocoran. Kedua, R4 menunjukkan bahwa sumber label
berpengaruh besar, tetapi tidak memakai dataset *crowdsourced* berskala besar. Penelitian ini mengisi kedua celah
tersebut dengan mengevaluasi ML klasik dan CNN ringan pada COUGHVID dengan prevalensi uji alami, metrik berbasis AUC,
dan perbandingan langsung label *self-reported* vs label ahli.
