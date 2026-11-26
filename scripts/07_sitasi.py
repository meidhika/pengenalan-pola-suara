"""Tahap 7a — penomoran sitasi gaya IEEE dan pembuatan Daftar Pustaka.

Draf bab ditulis dengan kunci sitasi `[@kunci]` (kunci = entri references/references.bib).
Script ini:
  1. menyimpan versi berkunci di reports/final/draft/_kunci/ (sumber yang diedit bila draf berubah);
  2. memberi nomor [1], [2], ... sesuai urutan kemunculan PERTAMA di bab 00 → 07;
  3. menulis bab bernomor ke reports/final/draft/ dan 06_daftar_pustaka.md.

Pemakaian:  python scripts/07_sitasi.py
Bila draf perlu diedit: edit file di draft/_kunci/, lalu jalankan ulang script ini.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "reports" / "final" / "draft"
KUNCI = DRAFT / "_kunci"
BAB = ["00_abstrak.md", "01_pendahuluan.md", "02_tinjauan_pustaka.md", "03_metodologi.md",
       "04_hasil_dan_pembahasan.md", "05_kesimpulan.md", "07_lampiran.md"]

# Entri IEEE (urutan di sini tidak berpengaruh; nomor ditentukan urutan sitasi di teks).
IEEE = {
    "xiong2022reliability": 'H. Xiong, S. Berkovsky, M. A. Kâafar, A. Jaffe, E. Coiera, and R. V. Sharan, "Reliability of crowdsourced data and patient-reported outcome measures in cough-based COVID-19 screening," *Scientific Reports*, vol. 12, Art. no. 21990, 2022, doi: 10.1038/s41598-022-26492-5.',
    "lella2021review": 'K. K. Lella and P. J. A. Alphonse, "A literature review on COVID-19 disease diagnosis from respiratory sound data," *AIMS Bioengineering*, vol. 8, no. 2, pp. 140–153, 2021, doi: 10.3934/bioeng.2021013.',
    "brown2020sounds": 'C. Brown, J. Chauhan, A. Grammenos, J. Han, A. Hasthanasombat, D. Spathis, T. Xia, P. Cicuta, and C. Mascolo, "Exploring automatic diagnosis of COVID-19 from crowdsourced respiratory sound data," in *Proc. 26th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining (KDD)*, 2020, pp. 3474–3484, doi: 10.1145/3394486.3412865.',
    "sharma2020coswara": 'N. Sharma, P. Krishnan, R. Kumar, S. Ramoji, S. R. Chetupalli, N. R., P. K. Ghosh, and S. Ganapathy, "Coswara — A database of breathing, cough, and voice sounds for COVID-19 diagnosis," in *Proc. Interspeech*, 2020, pp. 4811–4815.',
    "orlandic2021coughvid": 'L. Orlandic, T. Teijeiro, and D. Atienza, "The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms," *Scientific Data*, vol. 8, Art. no. 156, 2021, doi: 10.1038/s41597-021-00937-4.',
    "laguarta2020": 'J. Laguarta, F. Hueto, and B. Subirana, "COVID-19 artificial intelligence diagnosis using only cough recordings," *IEEE Open Journal of Engineering in Medicine and Biology*, vol. 1, pp. 275–281, 2020, doi: 10.1109/OJEMB.2020.3026928.',
    "lella2022dcnn": 'K. K. Lella and A. Pja, "Automatic diagnosis of COVID-19 disease using deep convolutional neural network with multi-feature channel from respiratory sound data: Cough, voice, and breath," *Alexandria Engineering Journal*, vol. 61, no. 2, pp. 1319–1334, 2022, doi: 10.1016/j.aej.2021.06.024.',
    "kranthikumar2022lwcnn": 'L. Kranthi Kumar and P. J. A. Alphonse, "COVID-19 disease diagnosis with light-weight CNN using modified MFCC and enhanced GFCC from human respiratory sounds," *The European Physical Journal Special Topics*, vol. 231, pp. 3329–3346, 2022, doi: 10.1140/epjs/s11734-022-00432-w.',
    "sela2026materi": 'E. I. Sela, "Pengenalan Pola Suara," materi kuliah Pertemuan 8, mata kuliah Pengenalan Pola dan Kecerdasan Buatan, Program Studi Magister Teknologi Informasi, 2026.',
    "oppenheim2010": 'A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed. Upper Saddle River, NJ, USA: Pearson, 2010.',
    "davis1980": 'S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 28, no. 4, pp. 357–366, 1980.',
    "oshaughnessy1987": "D. O'Shaughnessy, *Speech Communication: Human and Machine*. Reading, MA, USA: Addison-Wesley, 1987.",
    "shao2009gfcc": 'Y. Shao, Z. Jin, D. Wang, and S. Srinivasan, "An auditory-based feature for robust speech recognition," in *Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)*, 2009, pp. 4625–4628.',
    "mauch2014pyin": 'M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using probabilistic threshold distributions," in *Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)*, 2014, pp. 659–663.',
    "cortes1995": 'C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.',
    "breiman2001": 'L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.',
    "chen2016xgb": 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.',
    "lecun2015": 'Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436–444, 2015.',
    "ioffe2015": 'S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, 2015, pp. 448–456.',
    "sainburg2020": 'T. Sainburg, M. Thielk, and T. Q. Gentner, "Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires," *PLOS Computational Biology*, vol. 16, no. 10, Art. no. e1008228, 2020 (pustaka *noisereduce*).',
    "mcfee2015": 'B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg, and O. Nieto, "librosa: Audio and music signal analysis in Python," in *Proc. 14th Python in Science Conf. (SciPy)*, 2015, pp. 18–25.',
    "malek2023spafe": 'A. Malek, "spafe: Simplified python audio features extraction," *Journal of Open Source Software*, vol. 8, no. 81, Art. no. 4739, 2023.',
    "pedregosa2011": 'F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.',
    "lemaitre2017": 'G. Lemaître, F. Nogueira, and C. K. Aridas, "Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning," *Journal of Machine Learning Research*, vol. 18, no. 17, pp. 1–5, 2017.',
    "paszke2019": 'A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019, pp. 8024–8035.',
    "loshchilov2019": 'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.',
    "park2019specaug": 'D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le, "SpecAugment: A simple data augmentation method for automatic speech recognition," in *Proc. Interspeech*, 2019, pp. 2613–2617.',
    "chawla2002": 'N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.',
    "youden1950": 'W. J. Youden, "Index for rating diagnostic tests," *Cancer*, vol. 3, no. 1, pp. 32–35, 1950.',
    "saito2015": 'T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLOS ONE*, vol. 10, no. 3, Art. no. e0118432, 2015.',
    "efron1993": 'B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993.',
    "delong1988": 'E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach," *Biometrics*, vol. 44, no. 3, pp. 837–845, 1988.',
    "sun2014": "X. Sun and W. Xu, \"Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves,\" *IEEE Signal Processing Letters*, vol. 21, no. 11, pp. 1389–1393, 2014.",
    "cohen1960": 'J. Cohen, "A coefficient of agreement for nominal scales," *Educational and Psychological Measurement*, vol. 20, no. 1, pp. 37–46, 1960.',
    "mann1947": 'H. B. Mann and D. R. Whitney, "On a test of whether one of two random variables is stochastically larger than the other," *The Annals of Mathematical Statistics*, vol. 18, no. 1, pp. 50–60, 1947.',
    "benjamini1995": 'Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate: A practical and powerful approach to multiple testing," *Journal of the Royal Statistical Society: Series B*, vol. 57, no. 1, pp. 289–300, 1995.',
    "selvaraju2017": 'R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 618–626.',
    "vandermaaten2008": 'L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *Journal of Machine Learning Research*, vol. 9, pp. 2579–2605, 2008.',
}

POLA = re.compile(r"\[@([A-Za-z0-9_]+)\]")


def main():
    KUNCI.mkdir(exist_ok=True)
    for b in BAB:  # simpan versi berkunci sekali (bila belum ada)
        src, dst = DRAFT / b, KUNCI / b
        if not dst.exists():
            if not src.exists():
                sys.exit(f"Bab tidak ditemukan: {src}")
            shutil.copy2(src, dst)

    nomor: dict[str, int] = {}
    teks = {b: (KUNCI / b).read_text(encoding="utf-8") for b in BAB}
    for b in BAB:
        for k in POLA.findall(teks[b]):
            if k not in IEEE:
                sys.exit(f"Kunci tanpa entri IEEE: {k} (di {b})")
            nomor.setdefault(k, len(nomor) + 1)

    for b in BAB:
        out = POLA.sub(lambda m: f"[{nomor[m.group(1)]}]", teks[b])
        (DRAFT / b).write_text(out, encoding="utf-8")

    baris = ["# DAFTAR PUSTAKA", "",
             "Gaya IEEE; nomor sesuai urutan kemunculan pertama dalam teks.", ""]
    for k, n in sorted(nomor.items(), key=lambda kv: kv[1]):
        baris.append(f"[{n}] {IEEE[k]}")
        baris.append("")
    (DRAFT / "06_daftar_pustaka.md").write_text("\n".join(baris), encoding="utf-8")

    tak_dipakai = sorted(set(IEEE) - set(nomor))
    print(f"{len(nomor)} referensi disitasi; tidak disitasi: {tak_dipakai or '-'}")


if __name__ == "__main__":
    main()
