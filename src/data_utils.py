"""Utilitas data COUGHVID: memuat metadata dan memetakan uuid → file audio."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

AUDIO_EXT = (".webm", ".ogg", ".wav")
N_EXPERTS = 4
EXPERT_FIELDS = ["quality", "cough_type", "dyspnea", "wheezing", "stridor",
                 "choking", "congestion", "nothing", "diagnosis", "severity"]


def find_raw_dir(raw: Path | None = None) -> Path:
    """Folder hasil ekstrak yang berisi metadata_compiled.csv (mis. data/raw/coughvid_20211012)."""
    raw = Path(raw) if raw else load_config().paths.raw
    hits = sorted(raw.rglob("metadata_compiled.csv"))
    if not hits:
        raise FileNotFoundError(f"metadata_compiled.csv tidak ditemukan di {raw}. Jalankan scripts/01_extract.py")
    return hits[0].parent


def load_metadata(raw: Path | None = None, add_paths: bool = True) -> pd.DataFrame:
    """Muat metadata_compiled.csv dengan tipe kolom yang rapi.

    - Kolom indeks tanpa nama dibuang.
    - `cough_detected`, `age`, `latitude`, `longitude` → numerik.
    - `respiratory_condition`, `fever_muscle_pain` → boolean nullable.
    - Bila add_paths=True: tambah kolom `audio_path` dan `audio_ext` (NaN bila file tidak ada).
    """
    d = find_raw_dir(raw)
    df = pd.read_csv(d / "metadata_compiled.csv", low_memory=False)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == ""], errors="ignore")
    for c in ["cough_detected", "age", "latitude", "longitude"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["respiratory_condition", "fever_muscle_pain"]:
        if c in df:
            df[c] = df[c].map({True: True, False: False, "True": True, "False": False}).astype("boolean")
    if "datetime" in df:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    if add_paths:
        m = uuid_to_audio_path(d)
        df["audio_path"] = df["uuid"].map(m)
        df["audio_ext"] = df["audio_path"].map(lambda p: Path(p).suffix if isinstance(p, str) else np.nan)
    return df


def uuid_to_audio_path(raw_dir: Path | None = None) -> dict[str, str]:
    """Pemetaan uuid → path absolut file audio (.webm/.ogg/.wav) di folder raw."""
    raw_dir = Path(raw_dir) if raw_dir else find_raw_dir()
    out: dict[str, str] = {}
    for p in raw_dir.iterdir():
        if p.suffix.lower() in AUDIO_EXT:
            out[p.stem] = str(p)
    return out


def expert_label_columns(field: str = "diagnosis") -> list[str]:
    """Nama kolom label ahli, mis. diagnosis_1..diagnosis_4."""
    return [f"{field}_{i}" for i in range(1, N_EXPERTS + 1)]


def expert_labeled_mask(df: pd.DataFrame) -> pd.Series:
    """True bila rekaman punya minimal satu label diagnosis dari ahli."""
    return df[expert_label_columns("diagnosis")].notna().any(axis=1)


def probe_audio(path: str | Path) -> dict:
    """Info audio asli: codec, sample rate & kanal (ffprobe), durasi dari dekode penuh via ffmpeg.

    Durasi dihitung dari jumlah sampel hasil dekode karena header .webm sering tidak memuat durasi.
    """
    import json
    import subprocess

    path = str(path)
    pr = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels", "-of", "json", path],
        capture_output=True, text=True, check=True)
    st = json.loads(pr.stdout)["streams"][0]
    sr, ch = int(st["sample_rate"]), int(st["channels"])
    y = load_audio_ffmpeg(path, sr=sr, mono=True)
    return {"codec": st.get("codec_name"), "sr_native": sr, "channels": ch,
            "n_samples": len(y), "duration_s": len(y) / sr}


def load_audio_ffmpeg(path: str | Path, sr: int | None = 16000, mono: bool = True) -> np.ndarray:
    """Dekode file audio apa pun (webm/ogg/wav) via ffmpeg → array float32 [-1, 1]."""
    import subprocess

    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if mono:
        cmd += ["-ac", "1"]
    if sr:
        cmd += ["-ar", str(sr)]
    cmd += ["-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def expert_long(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah label ahli ke format panjang: satu baris per (uuid, expert)."""
    rows = []
    for i in range(1, N_EXPERTS + 1):
        cols = [f"{f}_{i}" for f in EXPERT_FIELDS if f"{f}_{i}" in df]
        sub = df[["uuid"] + cols].dropna(subset=[f"diagnosis_{i}"]).copy()
        sub.columns = ["uuid"] + [c.rsplit("_", 1)[0] for c in cols]
        sub["expert"] = i
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)
