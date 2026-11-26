# Klasifikasi Suara Batuk untuk Skrining COVID-19 Menggunakan Ekstraksi Fitur MFCC–GFCC dan Perbandingan *Machine Learning* Klasik dengan CNN Ringan pada Dataset COUGHVID

## ABSTRAK

Analisis suara batuk berpotensi menjadi alternatif skrining COVID-19 yang murah dan non-invasif, tetapi label
dataset *crowdsourced* umumnya bersifat *self-reported*. Penelitian ini membangun *pipeline* pengenalan pola suara
(akuisisi, *preprocessing*, ekstraksi fitur, klasifikasi, evaluasi) pada dataset COUGHVID v3. Rekaman dikonversi ke
16 kHz mono, di-*trim*, direduksi derau, dan dinormalisasi. Fitur yang diekstraksi meliputi 315 fitur tabular
(RMS, ZCR, spektral, *pitch*, MFCC±Δ, GFCC) serta *log-mel* dan MFCC 2D. Model yang dibandingkan adalah
*Logistic Regression*, SVM, k-NN, *Random Forest*, XGBoost, dan CNN ringan (97.521 parameter). Data dibagi per
rekaman sebelum *resampling* dan augmentasi, lalu diuji sekali pada prevalensi alami. Pada label *self-reported*
(COVID-19 vs sehat; test n = 1.624, prevalensi 6,7%), semua model hanya sedikit di atas acak. SVM mencapai ROC-AUC
0,576 [CI 95% 0,523–0,629] dan PR-AUC 0,088, sedangkan CNN *log-mel* 0,567 dan CNN MFCC 2D 0,544. Uji DeLong tidak
menemukan perbedaan bermakna antar-model. Dengan fitur dan *pipeline* yang sama, label diagnosis pulmonolog
menghasilkan ROC-AUC 0,826 [0,745–0,888]. Kedua sumber label tidak sepakat (Cohen's κ = 0,00). Disimpulkan bahwa
keandalan label, bukan arsitektur model, merupakan faktor pembatas utama, dan model berbasis label *self-reported*
belum layak untuk skrining.

**Kata kunci:** pengenalan pola suara, batuk, COVID-19, COUGHVID, MFCC, GFCC, CNN, keandalan label.

## *ABSTRACT*

*Cough sound analysis may offer an inexpensive, non-invasive alternative for COVID-19 screening, but labels in
crowdsourced datasets are usually self-reported. This study builds an audio pattern recognition pipeline
(acquisition, preprocessing, feature extraction, classification, evaluation) on the COUGHVID v3 dataset. Recordings
were converted to 16 kHz mono, trimmed, denoised, and normalised. The extracted features comprise 315 tabular
features (RMS, ZCR, spectral, pitch, MFCC±Δ, GFCC) together with log-mel and 2D MFCC representations. The models
compared are Logistic Regression, SVM, k-NN, Random Forest, XGBoost, and a lightweight CNN (97,521 parameters). Data
were split per recording before resampling and augmentation, and tested once at natural prevalence. With
self-reported labels (COVID-19 vs healthy; test n = 1,624, prevalence 6.7%), all models performed only slightly above
chance. The SVM reached a ROC-AUC of 0.576 [95% CI 0.523–0.629] and a PR-AUC of 0.088, while the log-mel CNN reached
0.567 and the 2D-MFCC CNN 0.544. DeLong tests found no significant differences between models. Using the same
features and pipeline, pulmonologist diagnosis labels yielded a ROC-AUC of 0.826 [0.745–0.888]. The two label sources
did not agree (Cohen's κ = 0.00). We conclude that label reliability, rather than model architecture, is the main
limiting factor, and that models trained on self-reported labels are not yet suitable for screening.*

***Keywords:** audio pattern recognition, cough, COVID-19, COUGHVID, MFCC, GFCC, CNN, label reliability.*
