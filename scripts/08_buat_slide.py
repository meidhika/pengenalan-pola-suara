"""Tahap 8 — membuat slide presentasi (python-pptx).

Keluaran: reports/final/Presentasi_COUGHVID_Kelompok.pptx (16:9, 15 slide, speaker notes Bahasa Indonesia).

Angka tabel hasil dibaca langsung dari reports/tables/06_tabel_final_{A,B}.csv agar konsisten dengan laporan;
angka lain diambil dari reports/ringkasan_hasil.md / PROGRESS.md (sumber: reports/tables/).

Jalankan:  python scripts/08_buat_slide.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
OUT = ROOT / "reports" / "final" / "Presentasi_COUGHVID_Kelompok.pptx"

# ---------------------------------------------------------------- tema
NAVY = RGBColor(0x10, 0x2A, 0x3C)      # warna dominan (judul, slide gelap)
NAVY2 = RGBColor(0x1C, 0x3F, 0x57)
ORANYE = RGBColor(0xD1, 0x66, 0x1F)    # = warna kelas COVID-19 di semua figur
BIRU = RGBColor(0x2A, 0x6F, 0xB0)      # = warna kelas healthy di semua figur
TINT = RGBColor(0xEE, 0xF3, 0xF7)      # latar kartu
TINT_OR = RGBColor(0xFB, 0xEE, 0xE4)
TEKS = RGBColor(0x1F, 0x29, 0x33)
ABU = RGBColor(0x5F, 0x6B, 0x76)
PUTIH = RGBColor(0xFF, 0xFF, 0xFF)
ES = RGBColor(0xC9, 0xDB, 0xEA)
F_JUDUL = "Cambria"
F_ISI = "Calibri"

W, H = 13.333, 7.5
M = 0.6                     # margin kiri/kanan
CW = W - 2 * M              # lebar area konten


def koma(x: float, d: int = 3) -> str:
    """Format angka gaya Indonesia (koma desimal)."""
    return f"{x:.{d}f}".replace(".", ",")


# ---------------------------------------------------------------- helper
def _font(run, size, color=TEKS, bold=False, italic=False, font=F_ISI):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font
    run.font.color.rgb = color


def _bullet(p, char="•", color=ORANYE, indent=0.22):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent))))
    pPr.set("indent", str(-int(Inches(indent))))
    for tag in ("a:buClr", "a:buChar", "a:buNone"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    clr = pPr.makeelement(qn("a:buClr"), {})
    s = clr.makeelement(qn("a:srgbClr"), {"val": str(color)})
    clr.append(s)
    pPr.append(clr)
    pPr.append(pPr.makeelement(qn("a:buChar"), {"char": char}))


def text(slide, x, y, w, h, paras, size=16, color=TEKS, bold=False, font=F_ISI, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, bullets=False, space_after=6, italic=False, line_spacing=None, name=None):
    """paras: str | list[str | list[(teks, opsi)]]. Opsi run: dict(bold, color, italic, size)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0)
    tf.margin_top = tf.margin_bottom = Inches(0)
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        if line_spacing:
            p.line_spacing = line_spacing
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            r = p.add_run()
            r.text = t
            _font(r, o.get("size", size), o.get("color", color), o.get("bold", bold), o.get("italic", italic),
                  o.get("font", font))
        if bullets:
            _bullet(p)
    return tb


def box(slide, x, y, w, h, fill=TINT, line=None, radius=0.08, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(radius / min(w, h), 0.5)
    s.text_frame.text = ""
    return s


def circle_num(slide, x, y, d, label, fill=ORANYE, color=PUTIH, size=14):
    s = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    _font(r, size, color, bold=True)
    return s


def picture(slide, name, x, y, w, h, align="center"):
    """Sisipkan gambar sebesar mungkin di dalam kotak (x, y, w, h) dengan rasio asli."""
    path = FIG / name
    iw, ih = Image.open(path).size
    r = iw / ih
    if w / h > r:
        ph, pw = h, h * r
    else:
        pw, ph = w, w / r
    px = x + (w - pw) / 2 if align == "center" else x
    py = y + (h - ph) / 2
    pic = slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
    pic.name = name
    return px, py, pw, ph


def stat(slide, x, y, w, h, big, label, color=ORANYE, fill=TINT, big_size=34, label_size=13):
    box(slide, x, y, w, h, fill=fill)
    text(slide, x + 0.25, y + 0.18, w - 0.5, 0.75, big, size=big_size, bold=True, color=color, font=F_JUDUL,
         anchor=MSO_ANCHOR.TOP, space_after=0)
    text(slide, x + 0.25, y + 0.18 + big_size / 72 * 1.25, w - 0.5, h - 0.3 - big_size / 72 * 1.25, label,
         size=label_size, color=TEKS, space_after=0)


def table(slide, x, y, w, rows, col_w, header_fill=NAVY, size=12, row_h=0.34, bold_rows=(), hl_rows=(),
          align_cols=None):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gt.table
    # matikan gaya bawaan agar warna kita yang dipakai
    tblPr = tbl._tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "1")
    tot = sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(w * cw / tot)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = header_fill
            elif i in hl_rows:
                c.fill.fore_color.rgb = TINT_OR
            else:
                c.fill.fore_color.rgb = PUTIH if i % 2 else TINT
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (align_cols[j] if align_cols else (PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER))
            r = p.add_run()
            r.text = str(val)
            _font(r, size, PUTIH if i == 0 else TEKS, bold=(i == 0 or i in bold_rows))
    return gt


def notes(slide, s: str):
    slide.notes_slide.notes_text_frame.text = " ".join(s.split())


# ---------------------------------------------------------------- kerangka slide
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
TOTAL = 15
_n = [0]


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color


def content_slide(tahap: str, judul: str):
    """Slide konten: label tahap (pill) + judul di posisi tetap + footer."""
    _n[0] += 1
    s = prs.slides.add_slide(BLANK)
    bg(s, PUTIH)
    # pill penanda tahap pipeline (motif berulang)
    pw = 0.16 + 0.085 * len(tahap)
    pill = box(s, M, 0.38, pw, 0.32, fill=TINT, radius=0.16, name="Label tahap")
    tf = pill.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = tahap.upper()
    _font(r, 10.5, NAVY2, bold=True)
    text(s, M, 0.78, CW, 0.75, judul, size=32, bold=True, color=NAVY, font=F_JUDUL, anchor=MSO_ANCHOR.TOP,
         name="Judul")
    text(s, M, 7.02, 8, 0.3, "Klasifikasi suara batuk COVID-19 · COUGHVID v3", size=10, color=ABU)
    text(s, W - M - 1.0, 7.02, 1.0, 0.3, f"{_n[0]} / {TOTAL}", size=10, color=ABU, align=PP_ALIGN.RIGHT)
    return s


def waveform_motif(slide, x, y, w, h, n=64, seed=3):
    """Motif gelombang batuk: batang vertikal dengan 'ledakan' energi (sinyal nonperiodik)."""
    import random
    rnd = random.Random(seed)
    bursts = [(0.10, 0.9), (0.28, 0.65), (0.47, 1.0), (0.70, 0.75), (0.86, 0.5)]
    bw = w / n
    for i in range(n):
        t = i / (n - 1)
        a = 0.06 + sum(amp * math.exp(-((t - c) / 0.045) ** 2) for c, amp in bursts)
        a = min(1.0, a * (0.75 + 0.5 * rnd.random()))
        bh = max(0.06, a * h)
        col = ORANYE if a > 0.55 else (BIRU if a > 0.2 else NAVY2)
        box(slide, x + i * bw + bw * 0.18, y + (h - bh) / 2, bw * 0.64, bh, fill=col, radius=bw * 0.3)


# =============================================================== 1. Judul
_n[0] += 1
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
text(s, M, 0.75, 9, 0.35, "TUGAS KELOMPOK · PENGENALAN POLA DAN KECERDASAN BUATAN", size=13, bold=True, color=ES)
text(s, M, 1.35, 8.2, 2.0, "Klasifikasi Suara Batuk untuk Skrining COVID-19", size=44, bold=True, color=PUTIH,
     font=F_JUDUL, line_spacing=0.95)
text(s, M, 3.25, 7.8, 0.9, "Ekstraksi fitur MFCC–GFCC serta perbandingan machine learning klasik dan CNN ringan "
     "pada dataset COUGHVID v3", size=18, color=ES)
ANGGOTA = [("Meidhika Nawa Sapta", "6250211011"), ("Ari Risnanto", "6250211006"),
           ("Roslilie Simbulan", "6250211001")]
anggota = [[("Disusun oleh", {"bold": True, "color": ORANYE, "size": 13})]] + \
          [[(f"{n} — {m}", {})] for n, m in ANGGOTA]
text(s, M, 4.55, 6, 1.6, anggota, size=16, color=PUTIH, space_after=3)
text(s, M, 6.45, 9, 0.4, "Dosen pengampu: Enny Itje Sela · Magister Teknologi Informasi · 2026", size=13, color=ES)
waveform_motif(s, 8.95, 1.6, 3.8, 3.6)
text(s, 8.95, 5.35, 3.8, 0.4, "rekaman batuk = sinyal nonperiodik", size=12, color=ES, italic=True,
     align=PP_ALIGN.CENTER)
notes(s, """
Assalamualaikum dan selamat pagi. Kami, Meidhika Nawa Sapta, Ari Risnanto, dan Roslilie Simbulan, akan mempresentasikan tugas kasus pengenalan
pola suara, yaitu klasifikasi suara batuk untuk skrining COVID-19 menggunakan dataset COUGHVID versi 3.
Penelitian ini mengikuti lima tahap pipeline pengenalan pola suara dari materi pertemuan delapan: akuisisi,
preprocessing, ekstraksi fitur, klasifikasi, dan evaluasi. Kami membandingkan model machine learning klasik pada
fitur MFCC dan GFCC dengan CNN ringan pada spektrogram. Pesan utama kami sejak awal: dengan protokol yang bebas
kebocoran data, suara batuk berlabel self-reported hanya sedikit lebih baik dari tebakan acak, dan kualitas label
ternyata menjadi faktor pembatas utama. Gambar di kanan adalah ilustrasi rekaman batuk: ledakan energi singkat yang
tidak periodik, berbeda dengan suara vokal. Presentasi berlangsung sekitar lima belas menit, lalu tanya jawab.
""")

# =============================================================== 2. Latar belakang
s = content_slide("Latar belakang", "Mengapa suara batuk untuk skrining COVID-19?")
text(s, M, 1.85, 6.7, 4.8, [
    [("Batuk ", {"bold": True}), ("adalah gejala utama COVID-19; tes PCR butuh biaya, tenaga, dan waktu.", {})],
    [("Skrining akustik ", {"bold": True}), ("via mikrofon smartphone/web: murah, cepat, dapat diakses luas.", {})],
    [("Batuk = sinyal nonperiodik ", {"bold": True}),
     ("(ledakan energi singkat, pita lebar) — bukan nada periodik seperti vokal.", {})],
    [("Klaim literatur tinggi ", {"bold": True}),
     ("(akurasi > 90%), tetapi dataset, label, dan protokol split berbeda.", {})],
    [("Label crowdsourced ", {"bold": True}), ("umumnya self-reported, tanpa konfirmasi PCR → noisy.", {})],
], size=17, bullets=True, space_after=12)
cx, cw = 7.75, CW - (7.75 - M)
stat(s, cx, 1.85, cw, 1.45, "34.434", "rekaman batuk crowdsourced di COUGHVID v3 (hingga Okt 2021)", color=NAVY)
stat(s, cx, 3.5, cw, 1.45, "91–95%", "akurasi dilaporkan R2/R3 pada dataset lain (Cambridge COVID-19 Sounds)",
     color=ORANYE)
stat(s, cx, 5.15, cw, 1.45, "0,83 vs 0,79", "AUC Random Forest R4: label PCR vs self-reported", color=BIRU)
notes(s, """
Batuk adalah salah satu gejala utama COVID-19. Tes PCR akurat, tetapi membutuhkan biaya, tenaga terlatih, dan waktu.
Karena itu muncul gagasan skrining akustik: orang cukup merekam batuk lewat smartphone atau browser, lalu model
mengklasifikasikannya. Dari sisi materi kuliah, batuk termasuk suara nonperiodik: berupa ledakan energi singkat
dengan spektrum lebar, berbeda dengan vokal yang periodik dan memiliki pitch jelas. Ini membuat pengenalannya lebih
sulit. Beberapa penelitian melaporkan akurasi di atas sembilan puluh persen, misalnya R2 dan R3, tetapi dengan dataset,
label, dan protokol pembagian data yang berbeda. Sebaliknya, R4 menunjukkan bahwa label self-reported kurang andal
dibanding label PCR. Dataset COUGHVID berisi lebih dari tiga puluh empat ribu rekaman crowdsourced dengan label yang
sebagian besar diisi sendiri oleh pengguna. Inilah titik berangkat penelitian kami.
""")

# =============================================================== 3. Rumusan masalah & tujuan
s = content_slide("Rumusan masalah & tujuan", "Empat pertanyaan, empat tujuan")
text(s, M + 0.65, 1.7, 5.4, 0.35, "RUMUSAN MASALAH", size=12, bold=True, color=ABU)
text(s, 7.15, 1.7, 5.4, 0.35, "TUJUAN", size=12, bold=True, color=ABU)
rm = [("Bagaimana merancang pipeline pengenalan pola batuk yang bebas kebocoran data?",
       "Pipeline 5 tahap reprodusibel (seed 42); split sebelum resampling, augmentasi, scaling."),
      ("Fitur akustik mana (MFCC, GFCC, spektral, waktu) yang paling informatif?",
       "Analisis fitur dengan uji Mann-Whitney + FDR dan ablasi kelompok fitur."),
      ("Bagaimana kinerja ML klasik dibanding CNN ringan pada data berprevalensi alami?",
       "Bandingkan ROC-AUC, PR-AUC, sens./spes. dengan CI bootstrap dan uji DeLong."),
      ("Sejauh mana keandalan label (self-reported vs ahli) memengaruhi hasil?",
       "Eksperimen B berlabel pulmonolog + uji silang sumber label.")]
y0, rh, gap = 2.12, 1.0, 0.18
for i, (q, t) in enumerate(rm):
    y = y0 + i * (rh + gap)
    circle_num(s, M, y + (rh - 0.48) / 2, 0.48, str(i + 1), fill=NAVY, size=16)
    box(s, M + 0.65, y, 5.3, rh, fill=TINT)
    text(s, M + 0.85, y, 4.9, rh, q, size=15, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    box(s, 6.7, y + rh / 2 - 0.16, 0.3, 0.32, fill=ORANYE, shape=MSO_SHAPE.RIGHT_ARROW)
    box(s, 7.15, y, CW - (7.15 - M), rh, fill=TINT_OR)
    text(s, 7.35, y, CW - (7.15 - M) - 0.4, rh, t, size=15, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
notes(s, """
Penelitian ini memiliki empat rumusan masalah yang masing-masing dipasangkan dengan satu tujuan. Pertama, bagaimana
merancang pipeline pengenalan pola suara batuk yang bebas kebocoran data; tujuannya membangun pipeline lima tahap yang
reprodusibel dengan seed 42, dengan pembagian data dilakukan sebelum undersampling, SMOTE, augmentasi, dan scaling.
Kedua, fitur mana yang paling informatif; kami menjawabnya dengan uji statistik Mann-Whitney yang dikoreksi FDR serta
ablasi kelompok fitur. Ketiga, bagaimana perbandingan model machine learning klasik dengan CNN ringan; kami mengukurnya
dengan ROC-AUC, PR-AUC, sensitivitas, dan spesifisitas pada data uji berprevalensi alami, lengkap dengan interval
kepercayaan dan uji DeLong. Keempat, seberapa besar pengaruh keandalan label; untuk ini kami membuat eksperimen B yang
memakai diagnosis pulmonolog, dan uji silang antar sumber label.
""")

# =============================================================== 4. Dataset
s = content_slide("1 · Akuisisi", "Dataset COUGHVID v3")
picture(s, "01_distribusi_label_demografi.png", M, 1.75, 7.6, 4.6)
text(s, M, 6.45, 7.6, 0.4, "Distribusi label self-reported dan demografi seluruh rekaman (n = 34.434).", size=11,
     color=ABU, italic=True)
cx, cw = 8.55, CW - (8.55 - M)
stat(s, cx, 1.75, cw, 1.4, "34.434", "rekaman webm/ogg/wav + JSON; 52 kolom metadata", color=NAVY, big_size=30)
stat(s, cx, 3.3, cw, 1.4, "720 : 10.132", "Eksp. A: COVID-19 vs healthy, cough_detected ≥ 0,8 (≈ 1 : 14)",
     color=ORANYE, big_size=30)
stat(s, cx, 4.85, cw, 1.4, "470 : 541", "Eksp. B: diagnosis pulmonolog (majority vote), setelah preprocessing",
     color=BIRU, big_size=30)
text(s, cx, 6.4, cw, 0.45, "Lisensi CC BY 4.0 · Orlandic, Teijeiro & Atienza (2021)", size=11, color=ABU)
notes(s, """
Tahap pertama adalah akuisisi. Kami memakai COUGHVID versi 3 dari Zenodo, berisi 34.434 rekaman batuk crowdsourced yang
direkam lewat aplikasi web dan smartphone hingga Oktober 2021. Formatnya webm, ogg, dan wav; menariknya sebagian file
berekstensi wav sebenarnya berisi codec AAC, sehingga semua file kami dekode dengan ffmpeg. Labelnya ada dua jenis.
Pertama, status self-reported: healthy, symptomatic, atau COVID-19. Kedua, label ahli dari hingga empat pulmonolog
untuk sekitar dua ribu delapan ratus rekaman. Mengikuti rekomendasi penulis dataset, kami hanya memakai rekaman dengan
probabilitas cough_detected minimal 0,8. Eksperimen A memakai label self-reported: 720 COVID-19 berbanding 10.132
healthy, sangat tidak seimbang, sekitar satu banding empat belas. Eksperimen B memakai majority vote diagnosis ahli:
470 COVID-19 dan 541 healthy cough. Dataset berlisensi CC BY 4.0 dan wajib disitasi.
""")

# =============================================================== 5. Alur penelitian
s = content_slide("Metodologi", "Alur penelitian mengikuti pipeline pengenalan pola suara")
picture(s, "07_alur_penelitian.png", M, 1.7, CW, 4.2)
chips = [("Split 70/15/15 per uuid", "stratified, seed 42 — dilakukan paling awal"),
         ("Train A di-undersample 3 : 1", "healthy : COVID-19, hanya pada train"),
         ("Val & test prevalensi alami", "≈ 1 : 14 → evaluasi jujur, test dipakai sekali")]
cw3 = (CW - 2 * 0.3) / 3
for i, (a, b) in enumerate(chips):
    x = M + i * (cw3 + 0.3)
    box(s, x, 6.05, cw3, 0.8, fill=TINT)
    text(s, x + 0.2, 6.1, cw3 - 0.4, 0.7, [[(a, {"bold": True, "color": NAVY})], [(b, {"size": 12, "color": ABU})]],
         size=14, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
notes(s, """
Ini adalah alur penelitian kami, yang sengaja mengikuti lima tahap pipeline pada slide sepuluh materi: akuisisi,
preprocessing, ekstraksi fitur, klasifikasi, dan evaluasi. Hal terpenting dari sisi metodologi adalah pencegahan
kebocoran data. Pembagian data train, validasi, dan test sebesar 70, 15, dan 15 persen dilakukan per rekaman atau uuid
secara stratified di awal, sebelum undersampling, SMOTE, augmentasi, scaling, dan seleksi fitur. Semua komponen
tersebut di-fit hanya pada train. Pada eksperimen A, train di-undersample menjadi tiga healthy untuk setiap satu
COVID-19 agar training di CPU tetap ringan, sementara validasi dan test dibiarkan berprevalensi alami sekitar satu
banding empat belas. Validasi dipakai untuk tuning ambang dan pemilihan model, dan test hanya dievaluasi sekali.
""")

# =============================================================== 6. Preprocessing
s = content_slide("2 · Preprocessing", "Preprocessing: dari rekaman mentah ke sinyal bersih")
picture(s, "02_langkah_preprocessing_covid.png", M, 1.65, 6.1, 5.2, align="left")
steps = [("Konversi", "ffmpeg → WAV mono 16 kHz, PCM 16-bit"),
         ("Trimming", "buang senyap awal/akhir (top_db = 30)"),
         ("Segmentasi", "deteksi batuk berbasis RMS (2× / 0,5× RMS global)"),
         ("Reduksi noise", "spectral gating, prop_decrease = 0,8"),
         ("Normalisasi", "peak → 0,99 (rentang −1..1)")]
x0 = 7.0
for i, (a, b) in enumerate(steps):
    y = 1.75 + i * 0.83
    circle_num(s, x0, y + 0.06, 0.42, "abcde"[i], fill=NAVY, size=14)
    text(s, x0 + 0.6, y, CW - (x0 - M) - 0.6, 0.75, [[(a, {"bold": True, "color": NAVY})], [(b, {})]], size=15,
         space_after=0)
box(s, x0, 6.0, CW - (x0 - M), 0.85, fill=TINT_OR)
text(s, x0 + 0.2, 6.0, CW - (x0 - M) - 0.4, 0.85,
     [[("5.864 file lolos", {"bold": True, "color": ORANYE}),
       (" · 17 dibuang (< 0,3 s setelah trim)", {})]],
     size=14, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
notes(s, """
Tahap kedua adalah preprocessing. Gambar di kiri menunjukkan satu contoh rekaman COVID-19 sebelum dan sesudah setiap
langkah. Pertama, semua format dikonversi dengan ffmpeg menjadi WAV mono 16 kilohertz dengan kuantisasi 16 bit. Sampling
rate 16 kilohertz berarti frekuensi Nyquist 8 kilohertz, sudah cukup untuk energi utama batuk. Kedua, bagian senyap di
awal dan akhir dibuang dengan ambang 30 desibel; contoh ini memendek dari 8,4 menjadi 4,3 detik. Ketiga, segmen batuk
dideteksi dari energi RMS per frame; kotak merah menunjukkan tiga segmen. Keempat, noise latar dikurangi dengan spectral
gating. Kelima, amplitudo dinormalisasi ke puncak 0,99 karena gain mikrofon tiap perangkat berbeda. Hasilnya 5.864 file
siap pakai dan hanya 17 yang dibuang karena terlalu pendek. Seluruh proses berjalan sekitar empat menit secara paralel.
""")

# =============================================================== 7. Representasi suara
s = content_slide("3 · Representasi suara", "Waveform, spectrogram, MFCC, dan GFCC")
picture(s, "03_representasi_waktu_frekuensi.png", M, 1.6, 5.3, 5.3, align="left")
xr = M + 5.6
wr = CW - 5.6
picture(s, "03_waveform_covid_vs_healthy.png", xr, 1.65, wr, 1.75)
konsep = [("Domain waktu", "waveform: amplitudo vs waktu; energi RMS & ZCR"),
          ("Fourier / STFT", "frame 32 ms, hop 10 ms, jendela Hann → spectrogram"),
          ("Log-mel & MFCC", "skala mel meniru telinga; DCT → 13 koefisien"),
          ("GFCC", "filter gammatone (40 pita) → 13 koefisien")]
gw = (wr - 0.3) / 2
for i, (a, b) in enumerate(konsep):
    x = xr + (i % 2) * (gw + 0.3)
    y = 3.65 + (i // 2) * 1.62
    box(s, x, y, gw, 1.42, fill=TINT)
    text(s, x + 0.2, y + 0.15, gw - 0.4, 1.15, [[(a, {"bold": True, "color": NAVY})], [(b, {})]], size=14,
         space_after=2)
notes(s, """
Tahap ketiga diawali dengan memahami representasi suara. Di kanan atas tampak waveform batuk COVID-19 dan healthy dalam
domain waktu: sumbu tegak adalah amplitudo. Polanya berupa ledakan energi yang tidak berulang secara teratur, sehingga
batuk termasuk suara nonperiodik dan pitch-nya tidak andal. Untuk melihat kandungan frekuensi kami memakai transformasi
Fourier jangka pendek atau STFT dengan frame 32 milidetik dan hop 10 milidetik, menghasilkan spectrogram di kiri atas.
Spectrogram kemudian dipetakan ke skala mel yang meniru persepsi telinga, menjadi log-mel spectrogram. MFCC diperoleh
dengan DCT dari log-mel, dan menggambarkan timbre atau bentuk selubung spektrum. GFCC serupa tetapi memakai filter
gammatone yang meniru koklea. Perlu dicatat, secara visual batuk COVID-19 dan healthy sulit dibedakan.
""")

# =============================================================== 8. Fitur & uji statistik
s = content_slide("3 · Ekstraksi fitur", "315 fitur tabular dan uji statistik per fitur")
groups = [("Domain waktu", "durasi, RMS, ZCR", 9), ("Spektral", "centroid, bandwidth, rolloff, flatness, contrast", 40),
          ("Pitch", "f0 (pYIN), voiced ratio", 6), ("MFCC", "13 koef. × 6 statistik", 78),
          ("MFCC Δ / ΔΔ", "dinamika temporal", 104), ("GFCC", "13 koef. × 6 statistik", 78)]
lw = 4.7
for i, (a, b, n) in enumerate(groups):
    y = 1.7 + i * 0.78
    box(s, M, y, lw, 0.66, fill=TINT)
    text(s, M + 0.2, y, lw - 1.2, 0.66, [[(a, {"bold": True, "color": NAVY, "size": 14})],
                                          [(b, {"size": 12, "color": ABU})]], anchor=MSO_ANCHOR.MIDDLE,
         space_after=0)
    text(s, M + lw - 0.95, y, 0.8, 0.66, str(n), size=20, bold=True, color=ORANYE, font=F_JUDUL,
         align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
text(s, M, 6.45, lw, 0.4, "+ fitur 2D untuk CNN: log-mel 64 pita & MFCC-2D 40 koef.", size=12, color=ABU,
     italic=True)
xr = M + lw + 0.4
wr = CW - lw - 0.4
sw = (wr - 0.3) / 2
stat(s, xr, 1.7, sw, 1.45, "39 / 315", "fitur signifikan di Eksp. A (Mann-Whitney, p_FDR < 0,05), efek kecil",
     color=ORANYE, big_size=30)
stat(s, xr + sw + 0.3, 1.7, sw, 1.45, "178 / 315", "fitur signifikan di Eksp. B (label ahli), efek sedang",
     color=BIRU, big_size=30)
picture(s, "03_distribusi_auc_fitur.png", xr, 3.45, 4.75, 4.75 / 1.711, align="left")
text(s, xr + 4.95, 3.6, wr - 4.95, 3.2, [
    [("Terkuat (A): ", {"bold": True}), ("contrast_05_std, AUC 0,584", {})],
    [("ZCR, RMS, centroid, f0: ", {"bold": True}), ("AUC 0,48–0,52", {})],
    [("PCA/t-SNE: ", {"bold": True}), ("kelas bercampur", {})],
], size=14, bullets=True, space_after=10)
notes(s, """
Dari setiap rekaman kami mengekstraksi 315 fitur tabular dalam enam kelompok: fitur domain waktu seperti durasi, energi
RMS, dan zero crossing rate; fitur spektral seperti centroid, rolloff, flatness, dan spectral contrast; pitch dengan
algoritma pYIN; MFCC beserta turunan delta dan delta-delta; serta GFCC. Setiap koefisien dirangkum dengan statistik
seperti rata-rata, standar deviasi, minimum, dan maksimum. Untuk CNN kami juga menyimpan log-mel dan MFCC dua dimensi.
Uji Mann-Whitney dengan koreksi FDR, hanya pada data train, menunjukkan perbedaan mencolok. Pada label self-reported
hanya 39 fitur yang signifikan dan semua efeknya kecil; fitur terkuat, spectral contrast pita tinggi, hanya mencapai AUC
0,584. Fitur klasik seperti ZCR dan centroid praktis setara acak. Sebaliknya, pada label ahli 178 fitur signifikan
dengan efek sedang. Ini petunjuk awal bahwa masalah utama ada pada label.
""")

# =============================================================== 9. Model
s = content_slide("4 · Klasifikasi", "Model: ML klasik dan LightCNN")
picture(s, "05_arsitektur.png", M, 1.65, CW, 2.3)
cw2 = (CW - 0.3) / 2
kartu = [("ML klasik — fitur tabular (315)", NAVY, [
    "LR, SVM RBF, k-NN, Random Forest, XGBoost + baseline dummy",
    "Pipeline: StandardScaler → SelectKBest → [SMOTE] → model",
    "Imbalance: class weight vs SMOTE (dipilih via CV train)",
    "Tuning: RandomizedSearchCV 5-fold, skor ROC-AUC"]),
    ("LightCNN — log-mel / MFCC-2D", ORANYE, [
        "4 blok konvolusi 3×3 (16–32–64–128) + BN, GAP; 97.521 parameter",
        "Augmentasi train: random crop 5,12 s, time shift, SpecAugment",
        "AdamW, pos_weight 3, early stopping pada AUC val",
        "3 seed (42, 7, 2024) → laporkan mean ± std"])]
for i, (judul, col, isi) in enumerate(kartu):
    x = M + i * (cw2 + 0.3)
    box(s, x, 4.15, cw2, 2.7, fill=TINT)
    text(s, x + 0.25, 4.3, cw2 - 0.5, 0.4, judul, size=16, bold=True, color=col)
    text(s, x + 0.25, 4.8, cw2 - 0.5, 2.0, isi, size=14, bullets=True, space_after=5)
notes(s, """
Tahap keempat adalah klasifikasi dengan dua keluarga model. Yang pertama machine learning klasik pada 315 fitur tabular:
Logistic Regression, SVM dengan kernel RBF, k-NN, Random Forest, dan XGBoost, ditambah baseline dummy yang menebak acak.
Semua model berada dalam satu pipeline: standardisasi, seleksi fitur SelectKBest, lalu opsional SMOTE, sehingga
semuanya di-fit di dalam fold training dan tidak bocor ke validasi. Hyperparameter dicari dengan RandomizedSearchCV
lima fold berbasis ROC-AUC. Yang kedua adalah CNN ringan atau LightCNN, terinspirasi R3, dengan empat blok konvolusi
tiga kali tiga, batch normalization, dan global average pooling; total hanya sekitar 97 ribu parameter sehingga dapat
dilatih di CPU. Inputnya log-mel atau MFCC dua dimensi, dengan augmentasi SpecAugment dan random crop. Ketidakseimbangan
kelas ditangani dengan pos_weight. Setiap konfigurasi dilatih dengan tiga seed.
""")

# =============================================================== 10. Hasil
s = content_slide("5 · Evaluasi", "Hasil Eksperimen A: AUC semua model hanya 0,54–0,58")
fa = pd.read_csv(TAB / "06_tabel_final_A.csv")
rows = [["Model", "ROC-AUC [CI 95%]", "PR-AUC", "Sens.", "Spes."]]
hl = []
for i, r in fa.iterrows():
    nm = r["model"]
    rows.append([nm, f"{koma(r.roc_auc)} [{koma(r.roc_auc_ci_low)}–{koma(r.roc_auc_ci_high)}]", koma(r.pr_auc),
                 koma(r.recall_sensitivity), koma(r.specificity)])
    if nm == "SVM (RBF)":
        hl.append(len(rows) - 1)
table(s, M, 1.7, 7.0, rows, [2.3, 2.3, 1.0, 0.9, 0.9], size=13, row_h=0.42, hl_rows=hl)
n_test = int(fa.n.iloc[0]); n_pos = int(fa.n_pos.iloc[0]); prev = fa.prevalensi.iloc[0]
text(s, M, 1.7 + 0.42 * len(rows) + 0.12, 7.0, 0.6,
     f"Test n = {n_test:,}".replace(",", ".") + f" ({n_pos} COVID-19, prevalensi {koma(prev)}); ambang Youden dari "
     f"val. Baris CNN = seed 42; mean 3 seed: log-mel 0,572 ± 0,006, MFCC 0,579 ± 0,030.",
     size=11, color=ABU, italic=True)
xr = M + 7.35
wr = CW - 7.35
picture(s, "06_roc_gabungan_A.png", xr, 1.7, wr, 2.3)
stat(s, xr, 4.2, wr, 1.25, "p = 0,12", "SVM vs dummy (DeLong): tidak bermakna; antarmodel p ≥ 0,39", color=ORANYE,
     big_size=28)
stat(s, xr, 5.6, wr, 1.25, "≈ 0,08", "precision vs prevalensi 0,067 → tidak layak untuk skrining", color=NAVY,
     big_size=28)
notes(s, """
Ini hasil utama pada data test eksperimen A, berisi 1.624 rekaman dengan hanya 108 COVID-19. Semua model, baik machine
learning klasik maupun CNN, memiliki ROC-AUC antara 0,54 dan 0,58. SVM RBF, yang terpilih sebagai model ML terbaik
berdasarkan data validasi, mencapai AUC 0,576 dengan interval kepercayaan 0,523 sampai 0,629. Batas bawahnya memang di
atas 0,5, tetapi uji DeLong terhadap baseline dummy memberi p sebesar 0,12, artinya tidak bermakna secara statistik.
Antarmodel juga tidak ada perbedaan bermakna; CNN tidak mengungguli ML klasik. PR-AUC sekitar 0,08 hingga 0,10, hampir
sama dengan prevalensi 0,067. Dalam praktik, dari sekitar dua belas rekaman yang diprediksi COVID-19 hanya satu yang
benar. Kurva ROC di kanan atas menempel dekat garis diagonal. Kesimpulannya, dengan label self-reported, model belum
layak dipakai untuk skrining.
""")

# =============================================================== 11. Keandalan label
s = content_slide("5 · Evaluasi", "Eksperimen B: kualitas label adalah faktor pembatas")
picture(s, "06_uji_silang_label.png", M, 1.7, 7.4, 3.6, align="left")
fb = pd.read_csv(TAB / "06_tabel_final_B.csv").set_index("model").loc["SVM (RBF)"]
box(s, M, 5.5, 7.4, 1.35, fill=TINT)
text(s, M + 0.25, 5.5, 6.9, 1.35, [
    [("S1: ", {"bold": True}), ("model label ahli 0,90 vs model self-reported 0,67 pada test berlabel ahli.", {})],
    [("Catatan: ", {"bold": True, "color": ORANYE}),
     ("label ahli = penilaian pulmonolog dari mendengar rekaman yang sama, bukan PCR.", {})]],
     size=14, anchor=MSO_ANCHOR.MIDDLE, space_after=6)
xr = M + 7.75
wr = CW - 7.75
stat(s, xr, 1.7, wr, 1.6, koma(fb.roc_auc),
     f"ROC-AUC SVM pada test B [{koma(fb.roc_auc_ci_low)}–{koma(fb.roc_auc_ci_high)}]; vs dummy p < 0,001",
     color=BIRU, big_size=36)
stat(s, xr, 3.45, wr, 1.6, "κ = 0,00", "kesepakatan self-reported vs ahli (n = 572; setuju 50,3%)", color=ORANYE,
     big_size=36)
stat(s, xr, 5.2, wr, 1.65, "43%", "rekaman yang mengaku COVID-19 juga dinilai COVID-19 oleh ahli", color=NAVY,
     big_size=36)
notes(s, """
Lalu mengapa hasil eksperimen A rendah? Eksperimen B menjawabnya. Dengan fitur dan pipeline yang sama persis, tetapi
memakai label diagnosis pulmonolog, SVM mencapai ROC-AUC 0,826, jauh di atas baseline dengan p kurang dari 0,001.
Kesepakatan antara label self-reported dan label ahli pada 572 rekaman hanya 50,3 persen dengan kappa Cohen nol, artinya
setara kebetulan. Hanya 43 persen rekaman yang mengaku COVID-19 juga dinilai COVID-19 oleh ahli. Grafik di kiri adalah
uji silang. Pada test berlabel ahli, model yang dilatih dengan label ahli mencapai 0,90, sedangkan model self-reported
0,67; jadi model A tetap menangkap sebagian pola akustik. Namun ada catatan penting: ahli mendiagnosis dengan mendengar
rekaman yang sama, sehingga label ahli konsisten secara akustik dan bukan kebenaran PCR. Temuan ini sejalan dengan R4.
""")

# =============================================================== 12. Analisis error
s = content_slide("5 · Evaluasi", "Analisis error dan subgrup (SVM, Eksperimen A)")
lw = 7.3
picture(s, "06_distribusi_skor_A.png", M, 1.65, lw, 2.0, align="left")
ins = [("Skor sempit & tumpang-tindih", "median healthy 0,242 vs COVID-19 0,254"),
       ("FP: rekaman panjang", "median 5,0 s vs TN 3,0 s; banyak batuk"),
       ("FN: rekaman sangat pendek", "0,8–1,0 s atau sedikit batuk"),
       ("Skor ~ durasi", "Spearman ρ: SVM 0,41; CNN log-mel 0,61")]
iw = (lw - 0.3) / 2
for i, (a, b) in enumerate(ins):
    x = M + (i % 2) * (iw + 0.3)
    y = 3.9 + (i // 2) * 1.5
    box(s, x, y, iw, 1.25, fill=TINT)
    text(s, x + 0.2, y, iw - 0.4, 1.25, [[(a, {"bold": True, "color": NAVY})], [(b, {})]], size=14,
         space_after=2, anchor=MSO_ANCHOR.MIDDLE)
xr = M + lw + 0.35
wr = CW - lw - 0.35
picture(s, "06_subgrup_svm_A.png", xr, 1.65, wr, 3.45)
box(s, xr, 5.4, wr, 1.25, fill=TINT_OR)
text(s, xr + 0.2, 5.4, wr - 0.4, 1.25, [
    [("Demam: ", {"bold": True}), ("AUC 0,51 vs 0,59", {})],
    [("Usia ≥ 55 th: ", {"bold": True}), ("AUC 0,48", {})],
    [("Gejala ber-confound dengan label; CI lebar", {"italic": True, "color": ABU, "size": 12})]],
     size=14, anchor=MSO_ANCHOR.MIDDLE, space_after=3)
notes(s, """
Analisis error memperlihatkan perilaku model. Distribusi skor SVM untuk kedua kelas sangat sempit dan tumpang-tindih:
median healthy 0,242 dan COVID-19 0,254. Karena itu ambang keputusan harus diambil dari data validasi, bukan 0,5.
False positive didominasi rekaman panjang dengan banyak batuk, median 5 detik dibanding 3 detik pada true negative.
False negative sering berupa rekaman sangat pendek. Skor juga berkorelasi dengan durasi, terutama pada CNN log-mel.
Artinya model sebagian merespons kepadatan atau panjang batuk, bukan pola khas COVID-19. Analisis subgrup di kanan
menunjukkan AUC lebih rendah pada orang yang demam, 0,51 dibanding 0,59, dan pada usia 55 tahun ke atas, 0,48. Jadi model
belum mampu membedakan COVID-19 dari batuk bergejala lain. Namun subgrup hanya berisi 12 hingga 73 kasus positif,
sehingga temuan ini bersifat indikatif.
""")

# =============================================================== 13. Perbandingan literatur
s = content_slide("Pembahasan", "Perbandingan dengan literatur")
rows = [["Aspek", "R2 Lella & Pja (2022)", "R3 Kranthi Kumar (2022)", "R4 Xiong dkk. (2022)", "Penelitian ini"],
        ["Dataset", "COVID-19 Sounds (batuk, napas, suara)", "COVID-19 Sounds", "Batuk call-centre, 698 rekaman",
         "COUGHVID v3, batuk saja"],
        ["Label", "Dari aplikasi (self-reported)", "Tidak dinyatakan", "PCR vs self-reported",
         "A: self-reported; B: ahli"],
        ["Model", "DAE + GFCC + IMFCC → DCNN", "M-MFCC / E-GFCC → CNN ringan", "openSMILE 6.373 fitur → RF",
         "315 fitur → SVM dll.; LightCNN"],
        ["Protokol", "k-fold, tidak per subjek", "5 folder, tidak per subjek", "Per subjek, test PCR",
         "Split per uuid dulu; test sekali"],
        ["Hasil", "Akurasi 95,45% (batuk 93,12%)", "Akurasi 91,32%", "AUC 0,83 (PCR) vs 0,79",
         "AUC A 0,576 · B 0,826"]]
table(s, M, 1.7, CW, rows, [1.1, 2.6, 2.4, 2.4, 2.5], size=13, row_h=0.62, bold_rows=(5,),
      align_cols=[PP_ALIGN.LEFT] * 5)
box(s, M, 5.75, CW, 1.1, fill=TINT_OR)
text(s, M + 0.3, 5.75, CW - 0.6, 1.1, [
    [("Tidak dapat dibandingkan langsung: ", {"bold": True, "color": ORANYE}),
     ("dataset, modalitas, label, dan metrik berbeda. Akurasi pada data 1 : 14 menyesatkan "
      "(menebak \"healthy\" saja ≈ 93%). Hasil A konsisten dengan R4: label crowdsourced kurang andal.", {})]],
     size=15, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
notes(s, """
Tabel ini membandingkan penelitian kami dengan tiga referensi utama. R2 dan R3 melaporkan akurasi 91 hingga 95 persen,
tetapi pada dataset Cambridge COVID-19 Sounds yang multimodal, dengan validasi k-fold yang tidak dipisah per subjek dan
augmentasi tanpa urutan split yang jelas, sehingga berisiko kebocoran data. Mereka juga melaporkan akurasi, bukan AUC.
Pada data kami yang sangat tidak seimbang, model yang selalu menebak healthy sudah mendapat akurasi sekitar 93 persen,
jadi akurasi bukan metrik yang tepat. R4 paling dekat dengan pendekatan kami: dengan pembagian per subjek, Random Forest
mencapai AUC 0,83 dengan label PCR dan 0,79 dengan label self-reported. Tidak ada referensi yang memakai COUGHVID, jadi
angka-angka ini tidak dapat dibandingkan langsung. Namun arah temuan kami konsisten dengan R4: label crowdsourced
self-reported membatasi kinerja klasifikasi.
""")

# =============================================================== 14. Kesimpulan & saran
s = content_slide("Penutup", "Kesimpulan dan saran")
cw2l = 7.0
text(s, M, 1.7, cw2l, 0.35, "KESIMPULAN", size=12, bold=True, color=ABU)
kes = [("Pipeline 5 tahap bebas leakage", "reprodusibel; split sebelum resampling/augmentasi/scaling"),
       ("MFCC (+Δ) & kontras spektral paling informatif", "fitur domain waktu (RMS, ZCR) paling lemah"),
       ("Eksp. A: AUC 0,54–0,58", "CNN tidak unggul dari ML klasik; tidak layak untuk skrining"),
       ("Label ahli: AUC 0,826; κ = 0,00", "kualitas label = faktor pembatas utama")]
for i, (a, b) in enumerate(kes):
    y = 2.12 + i * 1.18
    circle_num(s, M, y + 0.2, 0.48, str(i + 1), fill=NAVY, size=16)
    text(s, M + 0.7, y + 0.08, cw2l - 0.7, 1.0, [[(a, {"bold": True, "color": NAVY, "size": 16})], [(b, {})]],
         size=14, space_after=2)
xr = M + cw2l + 0.4
wr = CW - cw2l - 0.4
box(s, xr, 1.7, wr, 4.85, fill=TINT)
text(s, xr + 0.3, 1.9, wr - 0.6, 0.35, "SARAN", size=12, bold=True, color=ABU)
text(s, xr + 0.3, 2.45, wr - 0.6, 4.0, [
    "Validasi eksternal dengan label PCR (Coswara, COVID-19 Sounds)",
    "Fitur per segmen batuk / embedding pra-latih agar tidak bergantung durasi",
    "Kalibrasi probabilitas & ambang berbasis target sensitivitas",
    "Label noisy: status_SSL, label smoothing, co-teaching",
    "Gabungkan metadata gejala secara eksplisit"], size=16, bullets=True, space_after=14)
notes(s, """
Sebagai kesimpulan, pertama, kami berhasil membangun pipeline pengenalan pola suara batuk lima tahap yang reprodusibel
dan bebas kebocoran data. Kedua, fitur MFCC beserta turunannya dan spectral contrast paling informatif, sedangkan fitur
domain waktu paling lemah. Ketiga, pada label self-reported semua model hanya mencapai AUC 0,54 sampai 0,58, CNN ringan
tidak mengungguli machine learning klasik, dan model belum layak untuk skrining. Keempat, dengan label ahli AUC naik
menjadi 0,826, dan kesepakatan kedua label setara kebetulan; kualitas label adalah faktor pembatas utama. Saran kami:
validasi eksternal dengan data berlabel PCR, fitur per segmen batuk atau embedding pra-latih agar model tidak bergantung
pada durasi, kalibrasi probabilitas, teknik pelatihan yang tahan label noisy, dan pemanfaatan metadata gejala.
""")

# =============================================================== 15. Terima kasih
_n[0] += 1
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
text(s, M, 1.6, 7.5, 1.2, "Terima kasih", size=54, bold=True, color=PUTIH, font=F_JUDUL)
text(s, M, 2.75, 7.5, 0.6, "Pertanyaan & diskusi", size=24, color=ORANYE, bold=True)
text(s, M, 3.75, 7.3, 1.6, [
    [("Kode & reproduksi: ", {"bold": True, "color": PUTIH}),
     ("README.md — script 01→06, notebook, config.yaml (seed 42)", {})],
    [("Demo: ", {"bold": True, "color": PUTIH}), ("python scripts/predict.py <file_audio>", {})],
    [("Bukan alat diagnosis medis.", {"bold": True, "color": ORANYE})]], size=15, color=ES, space_after=8)
text(s, M, 5.9, 8.5, 0.9,
     "Dataset: Orlandic, L., Teijeiro, T., & Atienza, D. (2021). The COUGHVID crowdsourcing dataset. "
     "Scientific Data, 8, 156. doi:10.1038/s41597-021-00937-4 · CC BY 4.0", size=11, color=ES)
waveform_motif(s, 8.95, 1.6, 3.8, 3.6, seed=11)
notes(s, """
Demikian presentasi kami. Seluruh kode, notebook, dan konfigurasi tersedia beserta README yang menjelaskan cara
mereproduksi hasil, mulai dari script 01 hingga 06 dengan seed 42. Kami juga menyiapkan demo inferensi: satu file
audio batuk dapat diberikan ke script predict.py, yang menjalankan preprocessing dan ekstraksi fitur yang sama, lalu
mengeluarkan probabilitas dan label dari model terbaik. Perlu kami tegaskan bahwa demo ini hanya untuk tujuan
pembelajaran dan bukan alat diagnosis medis. Terima kasih kepada Ibu Enny Itje Sela atas bimbingannya, dan kepada
penulis dataset COUGHVID yang membagikan datanya dengan lisensi CC BY 4.0. Kami persilakan pertanyaan dan diskusi.
""")

assert _n[0] == TOTAL == len(prs.slides), (_n[0], len(prs.slides))
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"Tersimpan: {OUT.relative_to(ROOT)} ({len(prs.slides)} slide)")
if "--quiet" not in sys.argv:
    for i, sl in enumerate(prs.slides, 1):
        n = len(sl.notes_slide.notes_text_frame.text.split())
        print(f"  slide {i:2d}: {n} kata notes (±{n / 1.9:.0f} s pada ±115 kata/menit)")
