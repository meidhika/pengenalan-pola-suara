"""Tahap 1 — Inspeksi & ekstraksi dataset COUGHVID ke data/raw/.

Fitur:
- Inspeksi isi zip tanpa mengekstrak (namelist) + ringkasan jumlah file per ekstensi.
- Menangani zip bersarang (mis. public_dataset_v3.zip) secara rekursif.
- Idempoten: file yang sudah ada dengan ukuran sama dilewati.
- Ringkasan disimpan ke reports/logs/extract_summary.txt.

Pemakaian:
    .venv\\Scripts\\python.exe scripts/01_extract.py [--limit N] [--inspect-only]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import os
import shutil
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402

AUDIO_EXT = {".webm", ".ogg", ".wav"}


def summarize_zip(zf: zipfile.ZipFile) -> collections.Counter:
    """Hitung jumlah file per ekstensi (tanpa ekstrak)."""
    return collections.Counter(
        (os.path.splitext(i.filename)[1].lower() or "<dir>")
        for i in zf.infolist()
        if not i.is_dir()
    )


def _safe_target(dest: Path, member: str) -> Path:
    """Cegah path traversal (zip slip)."""
    target = (dest / member).resolve()
    if not str(target).startswith(str(dest.resolve())):
        raise ValueError(f"Path tidak aman di zip: {member}")
    return target


def extract_zip(zip_path: Path, dest: Path, limit: int | None, log: list[str]) -> dict:
    """Ekstrak zip ke dest (idempoten). Zip bersarang diekstrak rekursif ke foldernya sendiri."""
    stats = collections.Counter()
    with zipfile.ZipFile(zip_path) as zf:
        members = [i for i in zf.infolist() if not i.is_dir()]
        log.append(f"\n[ZIP] {zip_path.name}: {len(members)} file")
        for ext, n in sorted(summarize_zip(zf).items(), key=lambda x: -x[1]):
            log.append(f"    {ext:8s} {n}")
        if limit:
            members = members[:limit]
        nested = []
        for info in tqdm(members, desc=f"Ekstrak {zip_path.name[:30]}", unit="file"):
            target = _safe_target(dest, info.filename)
            if info.filename.lower().endswith(".zip"):
                nested.append((info, target))
            if target.exists() and target.stat().st_size == info.file_size:
                stats["skip"] += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            stats["extracted"] += 1
    # Zip bersarang → ekstrak ke folder bernama sama (tanpa .zip)
    for info, target in nested:
        log.append(f"  -> zip bersarang ditemukan: {info.filename}")
        sub = extract_zip(target, target.with_suffix(""), limit, log)
        stats.update({f"nested_{k}": v for k, v in sub.items()})
    return dict(stats)


def count_raw(raw: Path) -> tuple[collections.Counter, list[Path]]:
    """Hitung file hasil ekstrak per ekstensi dan cari metadata_compiled.csv."""
    c = collections.Counter()
    meta = []
    for p in raw.rglob("*"):
        if p.is_file():
            c[p.suffix.lower()] += 1
            if p.name == "metadata_compiled.csv":
                meta.append(p)
    return c, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="ekstrak N file pertama saja (uji cepat)")
    ap.add_argument("--inspect-only", action="store_true", help="hanya inspeksi, tanpa ekstrak")
    args = ap.parse_args()

    cfg = load_config()
    zip_path = cfg.paths.dataset_zip
    raw = cfg.paths.raw
    raw.mkdir(parents=True, exist_ok=True)
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)

    log = [f"Ringkasan ekstraksi COUGHVID — {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
           f"Sumber : {zip_path}", f"Tujuan : {raw}"]
    if args.inspect_only:
        with zipfile.ZipFile(zip_path) as zf:
            for ext, n in summarize_zip(zf).most_common():
                print(f"{ext:8s} {n}")
        return

    stats = extract_zip(zip_path, raw, args.limit, log)
    log.append(f"\nStatistik: {stats}")

    counts, meta = count_raw(raw)
    n_audio = sum(counts[e] for e in AUDIO_EXT)
    log.append("\nIsi data/raw/ setelah ekstraksi:")
    for ext, n in counts.most_common():
        log.append(f"    {ext:8s} {n}")
    log.append(f"Total file audio (.webm/.ogg/.wav): {n_audio}")
    log.append(f"metadata_compiled.csv: {[str(m.relative_to(cfg.paths.root)) for m in meta]}")

    out = cfg.paths.logs / "extract_summary.txt"
    out.write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log[-(len(counts) + 4):]))
    print(f"Ringkasan disimpan: {out}")


if __name__ == "__main__":
    main()
