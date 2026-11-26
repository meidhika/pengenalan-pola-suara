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


# ============================================================================
# Tahap 2 — penetapan populasi eksperimen, split, dan undersampling
# ============================================================================
LABEL_NAMES = {0: "healthy", 1: "COVID-19"}


def _majority_vote(values: list[str]) -> tuple[str | None, bool]:
    """Majority vote atas label beberapa ahli.

    Mengembalikan (label_mayoritas, seri). Bila dua label teratas berjumlah sama → (None, True).
    """
    if not values:
        return None, False
    vc = pd.Series(values).value_counts()
    if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
        return None, True
    return vc.index[0], False


def build_label_table(df: pd.DataFrame, experiment: str = "A", cfg=None,
                      return_stats: bool = False):
    """Tetapkan populasi dan label biner (1 = COVID-19, 0 = sehat) untuk satu eksperimen.

    Eksperimen
    ----------
    A : label *self-reported* `status` ∈ {COVID-19, healthy}, filter `cough_detected ≥ threshold`.
    B : label ahli `diagnosis_1..4`. Per rekaman dilakukan *majority vote* atas semua ahli yang
        memberi label; bila seri → rekaman dibuang. Rekaman dipakai bila hasil vote ∈
        {COVID-19, healthy_cough}. Opsional difilter `cough_detected ≥ threshold`
        (`data.expert_apply_cough_threshold`). Kolom `status` self-reported ikut disimpan untuk
        analisis keandalan label (perbandingan ala R4).
    C : label `status_SSL` ∈ {COVID-19, healthy} (hasil semi-supervised dari penulis dataset),
        filter `cough_detected ≥ threshold`.

    Parameter
    ---------
    df : DataFrame hasil `load_metadata()`.
    return_stats : bila True, kembalikan juga dict statistik (jumlah konflik, dll.).

    Return
    ------
    DataFrame kolom: uuid, label, label_name, cough_detected, status, audio_path (+ kolom ahli untuk B).
    """
    cfg = cfg or load_config()
    th = cfg.data.cough_detected_threshold
    exp = experiment.upper()
    stats: dict = {"experiment": exp}

    if exp in ("A", "C"):
        col = "status" if exp == "A" else "status_SSL"
        sub = df[(df["cough_detected"] >= th) & df[col].isin(["COVID-19", "healthy"])].copy()
        sub["label"] = (sub[col] == "COVID-19").astype(int)
    elif exp == "B":
        dcols = expert_label_columns("diagnosis")
        sub = df[expert_labeled_mask(df)].copy()
        votes = sub[dcols].apply(lambda r: [v for v in r if isinstance(v, str)], axis=1)
        res = votes.map(_majority_vote)
        sub["n_ahli"] = votes.map(len)
        sub["diagnosis_vote"] = res.map(lambda t: t[0])
        sub["vote_seri"] = res.map(lambda t: t[1])
        sub["ada_konflik"] = votes.map(lambda v: len(set(v)) > 1)
        sub["konflik_covid_vs_healthy"] = votes.map(lambda v: {"COVID-19", "healthy_cough"} <= set(v))
        stats.update({
            "berlabel_ahli": int(len(sub)),
            "multi_ahli": int((sub["n_ahli"] > 1).sum()),
            "konflik_antar_ahli": int(sub["ada_konflik"].sum()),
            "konflik_covid_vs_healthy": int(sub["konflik_covid_vs_healthy"].sum()),
            "dibuang_seri": int(sub["vote_seri"].sum()),
            "konflik_diselesaikan_vote": int((sub["ada_konflik"] & ~sub["vote_seri"]).sum()),
        })
        sub = sub[sub["diagnosis_vote"].isin(["COVID-19", "healthy_cough"])]
        stats["vote_covid_atau_healthy"] = int(len(sub))
        if cfg.data.get("expert_apply_cough_threshold", True):
            stats["dibuang_cough_detected"] = int((sub["cough_detected"] < th).sum())
            sub = sub[sub["cough_detected"] >= th]
        sub["label"] = (sub["diagnosis_vote"] == "COVID-19").astype(int)
    else:
        raise ValueError(f"Eksperimen tidak dikenal: {experiment}")

    sub["label_name"] = sub["label"].map(LABEL_NAMES)
    keep = ["uuid", "label", "label_name", "cough_detected", "status", "status_SSL", "audio_path"]
    if exp == "B":
        keep += ["diagnosis_vote", "n_ahli", "ada_konflik"]
    out = sub[keep].reset_index(drop=True)
    stats["n_covid"] = int(out["label"].sum())
    stats["n_healthy"] = int((out["label"] == 0).sum())
    return (out, stats) if return_stats else out


def make_splits(labels: pd.DataFrame, cfg=None, seed: int | None = None) -> pd.DataFrame:
    """Split STRATIFIED train/val/test per `uuid` (default 70/15/15, seed 42).

    Split dilakukan pada populasi asli (sebelum undersampling/augmentasi) agar val/test
    mempertahankan distribusi kelas alami → evaluasi jujur, tanpa *data leakage*.
    """
    from sklearn.model_selection import train_test_split

    cfg = cfg or load_config()
    seed = cfg.seed if seed is None else seed
    fr = cfg.split
    assert abs(fr.train + fr.val + fr.test - 1) < 1e-9, "Proporsi split harus berjumlah 1"
    assert labels["uuid"].is_unique, "uuid harus unik"
    tr, rest = train_test_split(labels, train_size=fr.train, stratify=labels["label"], random_state=seed)
    va, te = train_test_split(rest, test_size=fr.test / (fr.val + fr.test), stratify=rest["label"],
                              random_state=seed)
    out = labels.copy()
    out["split"] = ""
    out.loc[tr.index, "split"] = "train"
    out.loc[va.index, "split"] = "val"
    out.loc[te.index, "split"] = "test"
    return out


def undersample_train(splits: pd.DataFrame, ratio: float | None, seed: int = 42) -> pd.Series:
    """Tandai rekaman yang dipakai (kolom boolean `dipakai`).

    Undersampling acak kelas healthy (label 0) HANYA di train hingga healthy:COVID = ratio:1.
    Val/test selalu dipakai seluruhnya. ratio=None → semua dipakai.
    """
    used = pd.Series(True, index=splits.index)
    if ratio is None:
        return used
    tr = splits[splits["split"] == "train"]
    n_pos = int((tr["label"] == 1).sum())
    neg = tr[tr["label"] == 0]
    n_keep = min(len(neg), int(round(ratio * n_pos)))
    keep = neg.sample(n=n_keep, random_state=seed).index
    used.loc[neg.index.difference(keep)] = False
    return used
