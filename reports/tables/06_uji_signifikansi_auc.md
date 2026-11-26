Uji selisih ROC-AUC pada split test yang sama (DeLong; bootstrap berpasangan 2.000×).

| eksperimen | perbandingan | model_1 | model_2 | auc_1 | auc_2 | selisih | DeLong 95% CI | p DeLong | p bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| A | dua teratas (AUC test) | k-NN | SVM (RBF) | 0.580 | 0.576 | 0.004 | [-0.040; +0.047] | 0.867 | 0.857 |
| A | ML terbaik (val) vs CNN log-mel | SVM (RBF) | CNN (log-mel) | 0.576 | 0.567 | 0.009 | [-0.037; +0.055] | 0.696 | 0.719 |
| A | ML terbaik (val) vs CNN MFCC 2D | SVM (RBF) | CNN (MFCC 2D) | 0.576 | 0.544 | 0.032 | [-0.040; +0.104] | 0.388 | 0.402 |
| A | ML terbaik vs baseline | SVM (RBF) | Dummy (stratified) | 0.576 | 0.516 | 0.060 | [-0.015; +0.136] | 0.117 | 0.118 |
| B | dua teratas (AUC test) | SVM (RBF) | Random Forest | 0.826 | 0.817 | 0.009 | [-0.026; +0.044] | 0.616 | 0.625 |
| B | ML terbaik vs baseline | SVM (RBF) | Dummy (stratified) | 0.826 | 0.487 | 0.339 | [+0.229; +0.450] | 0.000 | 0.000 |
