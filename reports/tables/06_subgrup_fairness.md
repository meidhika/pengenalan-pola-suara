Performa per subgrup, Eksperimen A test (ambang Youden dari val). AUC hanya bila subgrup punya ≥ 5 positif dan ≥ 5 negatif. Keseluruhan: SVM (RBF) 0.576, CNN (log-mel) 0.567, CNN (MFCC 2D) 0.544

| variabel | subgrup | n | n_pos | prevalensi | AUC SVM [CI 95%] | Sens. SVM | Spes. SVM | AUC CNN (log-mel) | AUC CNN (MFCC 2D) |
|---|---|---|---|---|---|---|---|---|---|
| Demam / nyeri otot | tidak | 1472 | 73 | 0.050 | 0.590 [0.526–0.660] | 0.562 | 0.558 | 0.563 | 0.544 |
| Demam / nyeri otot | ya | 152 | 35 | 0.230 | 0.507 [0.395–0.613] | 0.543 | 0.496 | 0.532 | 0.541 |
| Gender | female | 537 | 50 | 0.093 | 0.585 [0.502–0.670] | 0.540 | 0.579 | 0.568 | 0.572 |
| Gender | male | 1077 | 55 | 0.051 | 0.565 [0.480–0.644] | 0.545 | 0.541 | 0.580 | 0.526 |
| Gender | other | 10 | 3 | 0.300 | – | 1.000 | 0.571 | – | – |
| Kelompok umur | < 25 | 289 | 18 | 0.062 | 0.565 [0.416–0.723] | 0.556 | 0.509 | 0.599 | 0.541 |
| Kelompok umur | 25–34 | 424 | 35 | 0.083 | 0.590 [0.479–0.694] | 0.629 | 0.542 | 0.562 | 0.559 |
| Kelompok umur | 35–44 | 388 | 17 | 0.044 | 0.631 [0.509–0.773] | 0.588 | 0.582 | 0.658 | 0.555 |
| Kelompok umur | 45–54 | 241 | 15 | 0.062 | 0.584 [0.430–0.733] | 0.533 | 0.540 | 0.553 | 0.521 |
| Kelompok umur | ≥ 55 | 195 | 12 | 0.062 | 0.475 [0.315–0.625] | 0.333 | 0.601 | 0.556 | 0.536 |
| Kelompok umur | tidak diisi | 87 | 11 | 0.126 | 0.542 [0.357–0.718] | 0.545 | 0.553 | 0.484 | 0.516 |
| Kondisi pernapasan | tidak | 1405 | 66 | 0.047 | 0.586 [0.515–0.656] | 0.530 | 0.559 | 0.546 | 0.530 |
| Kondisi pernapasan | ya | 219 | 42 | 0.192 | 0.529 [0.439–0.616] | 0.595 | 0.514 | 0.585 | 0.563 |
