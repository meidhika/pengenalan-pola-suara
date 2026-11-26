**Eksperimen B** — split test (n = 152, COVID-19 = 70, prevalensi = 0.461). Ambang = Youden's J yang dipilih di val. ★ = AUC val tertinggi (kriteria seleksi), ▲ = AUC test tertinggi (informasi).

| Model | Accuracy | Balanced Acc | Precision | Recall (Sens.) | Specificity | F1 | ROC-AUC [95% CI] | PR-AUC | AUC val | Ambang |
|---|---|---|---|---|---|---|---|---|---|---|
| Dummy (stratified) | 0.461 | 0.500 | 0.461 | 1.000 | 0.000 | 0.631 | 0.487 [0.417–0.567] | 0.454 | 0.428 | 0.000 |
| Logistic Regression | 0.796 | 0.794 | 0.783 | 0.771 | 0.817 | 0.777 | 0.812 [0.730–0.877] | 0.749 | 0.712 | 0.554 |
| SVM (RBF) ★ ▲ | 0.776 | 0.774 | 0.765 | 0.743 | 0.805 | 0.754 | 0.826 [0.745–0.888] | 0.786 | 0.718 | 0.557 |
| k-NN | 0.697 | 0.692 | 0.688 | 0.629 | 0.756 | 0.657 | 0.776 [0.689–0.848] | 0.675 | 0.648 | 0.539 |
| Random Forest | 0.763 | 0.766 | 0.718 | 0.800 | 0.732 | 0.757 | 0.817 [0.741–0.879] | 0.771 | 0.685 | 0.488 |
| XGBoost | 0.724 | 0.716 | 0.741 | 0.614 | 0.817 | 0.672 | 0.807 [0.726–0.871] | 0.754 | 0.713 | 0.586 |
