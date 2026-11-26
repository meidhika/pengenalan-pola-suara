"""Tahap 7b — Gabungkan draf bab (Markdown) menjadi laporan Word (.docx) + PDF.

Alur:
1. Buat ``reference.docx`` (Times New Roman 12, spasi 1,5, margin 3-3-3-3 cm, A4) dari
   reference bawaan pandoc, lalu atur gaya dengan python-docx.
2. Gabungkan ``reports/final/draft/00..07*.md`` + sampul + daftar isi/gambar/tabel (field TOC)
   menjadi satu Markdown, konversi dengan pandoc (pypandoc_binary) → rumus LaTeX menjadi OMML.
3. Pasca-proses python-docx: gaya caption, ukuran gambar (lebar ≤ 15 cm), format tabel,
   seksi & nomor halaman (sampul tanpa nomor, bagian awal romawi, isi arab), daftar pustaka.
4. (Opsional, Windows + MS Word) perbarui field TOC dan ekspor PDF via COM.

Contoh:
    python scripts/07b_buat_docx.py            # docx + pdf
    python scripts/07b_buat_docx.py --no-pdf   # docx saja (field TOC diperbarui saat dibuka di Word)
"""
from __future__ import annotations

import argparse

import re
import subprocess
from pathlib import Path

import pypandoc
from docx import Document

from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "reports" / "final" / "draft"
OUT_DIR = ROOT / "reports" / "final"
OUT_DOCX = OUT_DIR / "Laporan_COUGHVID_Kelompok.docx"
REF_DOCX = OUT_DIR / "reference.docx"
FONT = "Times New Roman"

JUDUL = ("Klasifikasi Suara Batuk untuk Skrining COVID-19 Menggunakan Ekstraksi Fitur MFCC–GFCC "
         "dan Perbandingan Machine Learning Klasik dengan CNN Ringan pada Dataset COUGHVID")

GAYA_GAMBAR = "Keterangan Gambar"
GAYA_TABEL = "Keterangan Tabel"
RE_CAPTION = re.compile(r"^(Gambar|Tabel)\s+L?\d+\.")
RE_PUSTAKA = re.compile(r"^\[\d+\]\s")


# ----------------------------------------------------------------------------- util XML
def set_font(style_or_run_font_el, name: str = FONT) -> None:
    """Set rFonts eksplisit (ascii/hAnsi/eastAsia/cs) dan hapus atribut tema."""
    rpr = style_or_run_font_el
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for att in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        if rfonts.get(qn(f"w:{att}")) is not None:
            del rfonts.attrib[qn(f"w:{att}")]
    for att in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{att}"), name)


def style_rpr(style):
    return style.element.get_or_add_rPr()


def raw_field(instr: str, placeholder: str) -> str:
    """Paragraf openxml berisi field (mis. TOC) yang ditandai dirty → diperbarui Word."""
    return (
        "```{=openxml}\n"
        "<w:p><w:r><w:fldChar w:fldCharType=\"begin\" w:dirty=\"true\"/></w:r>"
        f"<w:r><w:instrText xml:space=\"preserve\"> {instr} </w:instrText></w:r>"
        "<w:r><w:fldChar w:fldCharType=\"separate\"/></w:r>"
        f"<w:r><w:t>{placeholder}</w:t></w:r>"
        "<w:r><w:fldChar w:fldCharType=\"end\"/></w:r></w:p>\n"
        "```\n"
    )


RAW_SECTION_BREAK = (
    "```{=openxml}\n"
    "<w:p><w:pPr><w:sectPr><w:type w:val=\"nextPage\"/></w:sectPr></w:pPr></w:p>\n"
    "```\n"
)
RAW_PAGE_BREAK = "```{=openxml}\n<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>\n```\n"


# ----------------------------------------------------------------------------- 1. reference.docx
def buat_reference() -> None:
    subprocess.run([pypandoc.get_pandoc_path(), "-o", str(REF_DOCX),
                    "--print-default-data-file", "reference.docx"], check=True)
    doc = Document(str(REF_DOCX))
    st = doc.styles

    for s in st:
        if s.type in (WD_STYLE_TYPE.PARAGRAPH, WD_STYLE_TYPE.CHARACTER, WD_STYLE_TYPE.TABLE):
            if s.name in ("Source Code", "Verbatim Char") or s.name.endswith("Tok"):
                continue
            set_font(style_rpr(s))

    normal = st["Normal"]
    normal.font.size = Pt(12)
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before, pf.space_after = Pt(0), Pt(6)

    for nama in ("Body Text", "First Paragraph"):
        p = st[nama].paragraph_format
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.first_line_indent = Cm(1)
        p.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        p.space_before, p.space_after = Pt(0), Pt(6)
    st["Compact"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    st["Compact"].paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    st["Compact"].paragraph_format.space_after = Pt(2)

    ukuran = {1: 14, 2: 12, 3: 12}
    for lv in range(1, 10):
        h = st[f"Heading {lv}"]
        h.font.size = Pt(ukuran.get(lv, 12))
        h.font.bold = True
        h.font.italic = False
        h.font.color.rgb = RGBColor(0, 0, 0)
        hp = h.paragraph_format
        hp.keep_with_next = True
        hp.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        hp.first_line_indent = Cm(0)
        hp.space_before = Pt(12 if lv > 1 else 0)
        hp.space_after = Pt(6 if lv > 1 else 18)
    st["Heading 1"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    st["Heading 1"].paragraph_format.page_break_before = True

    # Gaya tambahan: caption, judul halaman depan, sampul
    def gaya_par(nama, size=12, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=6,
                 spasi=WD_LINE_SPACING.SINGLE, italic=False):
        s = st[nama] if nama in [x.name for x in st] else st.add_style(nama, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = normal
        set_font(style_rpr(s))
        s.font.size, s.font.bold, s.font.italic = Pt(size), bold, italic
        p = s.paragraph_format
        p.alignment, p.line_spacing_rule = align, spasi
        p.space_before, p.space_after = Pt(before), Pt(after)
        p.first_line_indent = Cm(0)
        return s

    gaya_par(GAYA_GAMBAR, size=11, before=4, after=12)
    gaya_par(GAYA_TABEL, size=11, before=12, after=4).paragraph_format.keep_with_next = True
    gaya_par("Judul Depan", size=14, bold=True, after=18, spasi=WD_LINE_SPACING.ONE_POINT_FIVE)
    gaya_par("Sampul Judul", size=16, bold=True, after=12, spasi=WD_LINE_SPACING.ONE_POINT_FIVE)
    gaya_par("Sampul Teks", size=12, after=0, spasi=WD_LINE_SPACING.ONE_POINT_FIVE)
    gaya_par("Sampul Tebal", size=12, bold=True, after=0, spasi=WD_LINE_SPACING.ONE_POINT_FIVE)
    gaya_par("Sampul Jarak", size=12, after=0)
    gaya_par("Isi Tabel", size=10, align=WD_ALIGN_PARAGRAPH.LEFT, after=0)
    gaya_par("Daftar Pustaka", size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6)
    dp = st["Daftar Pustaka"].paragraph_format
    dp.left_indent, dp.first_line_indent = Cm(0.9), Cm(-0.9)

    # Gaya TOC
    for lv in (1, 2, 3):
        nama = f"toc {lv}"
        s = st[nama] if nama in [x.name for x in st] else st.add_style(nama, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = normal
        s.paragraph_format.left_indent = Cm(0.6 * (lv - 1))
        s.paragraph_format.space_after = Pt(4)
        s.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        s.font.bold = lv == 1
    s = st["table of figures"] if "table of figures" in [x.name for x in st] else \
        st.add_style("table of figures", WD_STYLE_TYPE.PARAGRAPH)
    s.base_style = normal
    s.paragraph_format.left_indent, s.paragraph_format.first_line_indent = Cm(2.2), Cm(-2.2)
    s.paragraph_format.space_after = Pt(6)
    s.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

    # Kode sumber lebih kecil
    if "Source Code" not in [x.name for x in st]:
        sc = st.add_style("Source Code", WD_STYLE_TYPE.PARAGRAPH)
        sc.base_style = normal
        set_font(style_rpr(sc), "Consolas")
    sc = st["Source Code"]
    sc.font.size = Pt(9)
    sc.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    sc.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(sec, m, Cm(3))
    doc.save(str(REF_DOCX))


# ----------------------------------------------------------------------------- 2. Markdown gabungan
def div(gaya: str, teks: str) -> str:
    return f'::: {{custom-style="{gaya}"}}\n{teks}\n:::\n'


def sampul() -> str:
    bagian = [
        div("Sampul Tebal", "LAPORAN TUGAS KELOMPOK"),
        div("Sampul Teks", "Mata Kuliah Pengenalan Pola dan Kecerdasan Buatan"),
        div("Sampul Jarak", "&nbsp;"),
        div("Sampul Judul", JUDUL.replace("Machine Learning", "*Machine Learning*")),
        div("Sampul Jarak", "&nbsp;\n\n&nbsp;"),
        div("Sampul Teks", "Dosen Pengampu:"),
        div("Sampul Tebal", "Enny Itje Sela"),
        div("Sampul Jarak", "&nbsp;"),
        div("Sampul Teks", "Disusun oleh:"),
        div("Sampul Tebal", "[NAMA ANGGOTA] — [NIM]\\\n[NAMA ANGGOTA] — [NIM]\\\n[NAMA ANGGOTA] — [NIM]"),
        div("Sampul Jarak", "&nbsp;\n\n&nbsp;\n\n&nbsp;"),
        div("Sampul Tebal", "PROGRAM STUDI MAGISTER TEKNOLOGI INFORMASI"),
        div("Sampul Tebal", "2026"),
    ]
    return "\n".join(bagian)


def baca_bab(nama: str) -> str:
    teks = (DRAFT / nama).read_text(encoding="utf-8")
    teks = re.sub(r"<!--.*?-->", "", teks, flags=re.S)
    if nama.startswith("00_"):
        teks = re.sub(r"\A# .*\n", "", teks)              # judul sudah ada di sampul
        teks = teks.replace("## ABSTRAK", "# ABSTRAK").replace("## *ABSTRACT*", "# *ABSTRACT*")
    if nama.startswith("06_"):
        teks = re.sub(r"^Gaya IEEE;.*\n", "", teks, flags=re.M)
    return teks.strip() + "\n"


def gabung_markdown() -> str:
    bab = sorted(p.name for p in DRAFT.glob("0*.md"))
    awal, isi = [b for b in bab if b.startswith("00_")], [b for b in bab if not b.startswith("00_")]
    md = [sampul(), RAW_SECTION_BREAK]
    md += [baca_bab(b) for b in awal]
    md += [RAW_PAGE_BREAK, div("Judul Depan", "DAFTAR ISI"),
           raw_field('TOC \\o "1-3" \\h \\z \\u', "[Daftar isi — perbarui field (F9) di Word]"),
           RAW_PAGE_BREAK, div("Judul Depan", "DAFTAR GAMBAR"),
           raw_field(f'TOC \\h \\z \\t "{GAYA_GAMBAR},1"', "[Daftar gambar — perbarui field (F9)]"),
           RAW_PAGE_BREAK, div("Judul Depan", "DAFTAR TABEL"),
           raw_field(f'TOC \\h \\z \\t "{GAYA_TABEL},1"', "[Daftar tabel — perbarui field (F9)]"),
           RAW_SECTION_BREAK]
    md += [baca_bab(b) for b in isi]
    return "\n\n".join(md)


def jalankan_pandoc(md: str, out: Path) -> None:
    tmp = OUT_DIR / "_gabungan.md"
    tmp.write_text(md, encoding="utf-8")
    subprocess.run([
        pypandoc.get_pandoc_path(), str(tmp), "-o", str(out),
        "-f", "markdown-implicit_figures-auto_identifiers",
        "--reference-doc", str(REF_DOCX),
        "--resource-path", str(DRAFT),
        "--columns", "100000",          # tabel pipa tanpa lebar kolom tetap → autofit di Word
        "--dpi", "200",
    ], check=True)


# ----------------------------------------------------------------------------- 3. pasca-proses
def set_cell_borders(table) -> None:
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for sisi in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{sisi}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    lama = tblPr.find(qn("w:tblBorders"))
    if lama is not None:
        tblPr.remove(lama)
    tblPr.append(borders)
    # autofit
    layout = tblPr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tblPr.append(layout)
    layout.set(qn("w:type"), "autofit")
    # lebar tabel = 100% lebar teks
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    tblW.set(qn("w:type"), "pct")
    tblW.set(qn("w:w"), "5000")


def format_tabel(doc) -> int:
    n = 0
    for t in doc.tables:
        n += 1
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_cell_borders(t)
        for i, row in enumerate(t.rows):
            if i == 0:
                trPr = row._tr.get_or_add_trPr()
                h = OxmlElement("w:tblHeader")
                trPr.append(h)
            for cell in row.cells:
                for tcW in cell._tc.tcPr.findall(qn("w:tcW")) if cell._tc.tcPr is not None else []:
                    tcW.set(qn("w:type"), "auto")
                    tcW.set(qn("w:w"), "0")
                if i == 0:
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "D9D9D9")
                    cell._tc.get_or_add_tcPr().append(shd)
                for p in cell.paragraphs:
                    p.style = doc.styles["Isi Tabel"]
                    for r in p.runs:
                        r.font.size = Pt(10)          # termasuk teks kode (Verbatim Char)
                        if i == 0:
                            r.font.bold = True
    return n


def format_paragraf(doc) -> dict:
    stat = {"gambar": 0, "cap_gambar": 0, "cap_tabel": 0, "pustaka": 0}
    maks_w, maks_h = Cm(15), Cm(20)
    di_pustaka = False
    for p in doc.paragraphs:
        teks = p.text.strip()
        if p.style.name.startswith("Heading 1"):
            di_pustaka = teks.upper().startswith("DAFTAR PUSTAKA")
        # gambar
        drawings = p._p.findall(".//" + qn("wp:inline"))
        if drawings:
            for inl in drawings:
                ext = inl.find(qn("wp:extent"))
                cx, cy = int(ext.get("cx")), int(ext.get("cy"))
                skala = min(1.0, maks_w / cx, maks_h / cy)
                cx, cy = int(cx * skala), int(cy * skala)
                ext.set("cx", str(cx)); ext.set("cy", str(cy))
                for a_ext in inl.iter(qn("a:ext")):
                    a_ext.set("cx", str(cx)); a_ext.set("cy", str(cy))
                stat["gambar"] += 1
            pf = p.paragraph_format
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.first_line_indent = Cm(0)
            pf.keep_with_next = True
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            pf.space_before = Pt(6)
            continue
        m = RE_CAPTION.match(teks)
        if m and p.runs and p.runs[0].bold:
            if m.group(1) == "Gambar":
                p.style = doc.styles[GAYA_GAMBAR]; stat["cap_gambar"] += 1
            else:
                p.style = doc.styles[GAYA_TABEL]; stat["cap_tabel"] += 1
            continue
        if di_pustaka and RE_PUSTAKA.match(teks):
            p.style = doc.styles["Daftar Pustaka"]; stat["pustaka"] += 1
    return stat


def tambah_nomor_halaman(paragraph) -> None:
    run = paragraph.add_run()
    for tipe, instr in (("begin", None), (None, " PAGE "), ("separate", None), ("end", None)):
        if tipe:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tipe)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = instr
        run._r.append(el)


def atur_seksi(doc) -> int:
    secs = doc.sections
    for s in secs:
        s.page_width, s.page_height = Cm(21), Cm(29.7)
        for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
            setattr(s, m, Cm(3))
        s.footer_distance = Cm(1.5)
    format_nomor = {1: "lowerRoman", 2: "decimal"}
    for i, s in enumerate(secs):
        s.footer.is_linked_to_previous = False
        ft = s.footer.paragraphs[0] if s.footer.paragraphs else s.footer.add_paragraph()
        for r in list(ft.runs):
            r._r.getparent().remove(r._r)
        if i == 0:
            continue                                   # sampul tanpa nomor
        ft.alignment = WD_ALIGN_PARAGRAPH.CENTER
        tambah_nomor_halaman(ft)
        sectPr = s._sectPr
        pg = sectPr.find(qn("w:pgNumType"))
        if pg is None:
            pg = OxmlElement("w:pgNumType"); sectPr.append(pg)
        pg.set(qn("w:fmt"), format_nomor.get(i, "decimal"))
        pg.set(qn("w:start"), "1")
    return len(secs)


def pasca_proses(path: Path) -> dict:
    doc = Document(str(path))
    stat = format_paragraf(doc)
    stat["tabel"] = format_tabel(doc)
    stat["seksi"] = atur_seksi(doc)
    cp = doc.core_properties
    cp.title, cp.subject = JUDUL, "Pengenalan Pola dan Kecerdasan Buatan"
    cp.author, cp.language = "Kelompok COUGHVID", "id-ID"
    doc.save(str(path))
    return stat


# ----------------------------------------------------------------------------- 4. Word COM
def perbarui_dan_pdf(path: Path, buat_pdf: bool) -> dict:
    import win32com.client  # pywin32 (terpasang bersama docx2pdf)
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    info = {}
    try:
        doc = word.Documents.Open(str(path))
        # Switch \t "Gaya,level" memakai pemisah daftar sesuai locale (Indonesia: ";")
        sep = word.International(17)                     # wdListSeparator
        info["pemisah_locale"] = sep
        for f in doc.Fields:
            kode = f.Code.Text
            if f.Type == 13 and "\\t " in kode and ',1"' in kode and sep != ",":
                f.Code.Text = kode.replace(',1"', f'{sep}1"')
        doc.Fields.Update()
        for i in range(1, doc.TablesOfContents.Count + 1):
            doc.TablesOfContents(i).Update()
        doc.Repaginate()
        for i in range(1, doc.TablesOfContents.Count + 1):      # nomor halaman final
            doc.TablesOfContents(i).UpdatePageNumbers()
        info["halaman"] = doc.ComputeStatistics(2)
        info["toc_entri"] = [doc.TablesOfContents(i).Range.Paragraphs.Count
                             for i in range(1, doc.TablesOfContents.Count + 1)]
        info["gambar_inline"] = doc.InlineShapes.Count
        doc.Save()
        if buat_pdf:
            doc.ExportAsFixedFormat(str(path.with_suffix(".pdf")), 17,  # wdExportFormatPDF
                                    False, 0, 0, 1, 1, 0, True, True, 1)  # bookmark dari heading
        doc.Close(False)
    finally:
        word.Quit()
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-pdf", action="store_true", help="lewati ekspor PDF")
    ap.add_argument("--no-word", action="store_true", help="jangan panggil MS Word (TOC tidak diperbarui)")
    args = ap.parse_args()

    buat_reference()
    jalankan_pandoc(gabung_markdown(), OUT_DOCX)
    stat = pasca_proses(OUT_DOCX)
    print("pasca-proses:", stat)
    if not args.no_word:
        print("Word:", perbarui_dan_pdf(OUT_DOCX, not args.no_pdf))
    print("selesai:", OUT_DOCX)


if __name__ == "__main__":
    main()
