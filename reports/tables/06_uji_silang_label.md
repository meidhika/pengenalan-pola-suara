ROC-AUC [CI 95%] uji silang. Kolom = set evaluasi ← sumber label training. S1 n = 102 (51 COVID-19 ahli); S2 n = 1554 (79 COVID-19 self-reported); S3 n = 251 (prediksi identik, dua label). uuid train/val model sumber dibuang dari set evaluasi.

| model | S1: test B (label ahli) ← label ahli (B) | S1: test B (label ahli) ← label self-reported (A) | S2: test A (label self-reported) ← label ahli (B) | S2: test A (label self-reported) ← label self-reported (A) | S3: rekaman B tak terlihat model A — label ahli ← label self-reported (A) | S3: rekaman B tak terlihat model A — label self-reported ← label self-reported (A) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.874 [0.797–0.946] | 0.682 [0.576–0.784] | 0.559 [0.500–0.619] | 0.580 [0.515–0.645] | 0.596 [0.528–0.669] | 0.509 [0.391–0.616] |
| SVM (RBF) | 0.896 [0.825–0.960] | 0.673 [0.569–0.779] | 0.545 [0.487–0.604] | 0.585 [0.523–0.647] | 0.574 [0.504–0.648] | 0.541 [0.414–0.646] |
| k-NN | 0.840 [0.753–0.917] | 0.611 [0.498–0.721] | 0.574 [0.513–0.638] | 0.576 [0.516–0.638] | 0.517 [0.442–0.595] | 0.551 [0.437–0.660] |
| Random Forest | 0.887 [0.810–0.952] | 0.711 [0.607–0.815] | 0.561 [0.500–0.623] | 0.583 [0.521–0.648] | 0.616 [0.550–0.682] | 0.494 [0.375–0.593] |
| XGBoost | 0.874 [0.796–0.944] | 0.672 [0.569–0.768] | 0.578 [0.518–0.641] | 0.571 [0.505–0.639] | 0.590 [0.524–0.660] | 0.531 [0.420–0.640] |
| CNN (log-mel) | – | 0.653 [0.541–0.763] | – | – | 0.652 [0.588–0.719] | 0.489 [0.376–0.600] |
| CNN (MFCC 2D) | – | 0.489 [0.377–0.612] | – | – | 0.518 [0.450–0.590] | 0.516 [0.403–0.635] |
