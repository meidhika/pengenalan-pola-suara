"""Tahap 5 — CNN ringan (PyTorch) untuk fitur 2D (log-mel spectrogram & MFCC 2D).

Isi:
  - get_device(), default_batch_size()      : pilih CUDA bila ada, CPU bila tidak
  - load_split()                            : baca .npy (memmap) + rata-rata waktu (time pooling) → RAM
  - content_length()                        : jumlah frame berisi suara (sisanya padding konstan)
  - BandNormalizer                          : standardisasi per pita frekuensi (fit di TRAIN saja)
  - SpecDataset                             : Dataset + augmentasi HANYA untuk train
                                              (random crop, random time shift, SpecAugment)
  - mixup_batch()                           : mixup opsional (level batch)
  - LightCNN                                : 4–5 blok [Conv2d → BN → ReLU → MaxPool → Dropout] → GAP → FC(1)
                                              kernel 3×3 atau 1×12 (ala R3)
  - CRNN                                    : pembanding kecil (blok CNN + GRU dua arah)
  - count_params(), layer_table()           : jumlah parameter & tabel layer
  - predict_proba()                         : skor sigmoid untuk satu set data
  - grad_cam()                              : peta Grad-CAM pada blok konvolusi terakhir

Konvensi array: (N, n_pita, n_frame), pita frekuensi di sumbu tinggi (H), waktu di sumbu lebar (W).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.config import load_config

FITUR_2D = {"logmel": "log-mel", "mfcc2d": "MFCC 2D"}


# ---------------------------------------------------------------------------
# Perangkat
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_batch_size(device: torch.device) -> int:
    """GPU sanggup batch lebih besar; di CPU 32 memberi trade-off waktu/stabilitas BN yang baik."""
    return 64 if device.type == "cuda" else 32


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def content_length(x: np.ndarray, tol: float = 1e-3) -> int:
    """Jumlah frame berisi suara = sampai frame terakhir yang BERBEDA dari frame paling akhir.

    Sinyal dipad nol di akhir pada Tahap 3 → semua frame padding identik dengan frame terakhir
    (log-mel: puncak − 80 dB di semua pita; MFCC: vektor konstan). Dipakai agar random crop
    jatuh di area suara. Rekaman yang mengisi penuh 10 s → ≈ seluruh panjang.
    """
    diff = np.abs(x - x[:, -1:]).max(axis=0)
    idx = np.flatnonzero(diff > tol)
    return int(idx[-1] + 2) if len(idx) else x.shape[1]


def load_split(feature: str, experiment: str, split: str, time_pool: int = 2, limit_idx=None,
               mmap: bool = True):
    """Baca array fitur 2D + label + uuid; opsional rata-rata tiap `time_pool` frame (hemat komputasi).

    Return: X (N, F, T') float16, y (N,) int64, uuids (N,), lengths (N,) frame berisi suara (setelah pooling).
    """
    d = load_config().paths.features
    X = np.load(d / f"{feature}_{experiment}_{split}.npy", mmap_mode="r" if mmap else None)
    y = np.load(d / f"labels_{experiment}_{split}.npy").astype(np.int64)
    u = np.load(d / f"uuids_{experiment}_{split}.npy", allow_pickle=True)
    idx = np.arange(len(y)) if limit_idx is None else np.asarray(limit_idx)
    out, lens = [], []
    for s in range(0, len(idx), 256):  # baca bertahap dari memmap
        blk = np.asarray(X[idx[s:s + 256]], dtype=np.float32)
        lens += [content_length(b) for b in blk]
        if time_pool > 1:
            T = blk.shape[2] // time_pool * time_pool
            blk = blk[:, :, :T].reshape(blk.shape[0], blk.shape[1], -1, time_pool).mean(-1)
        out.append(blk.astype(np.float16))
    lens = np.ceil(np.asarray(lens) / max(time_pool, 1)).astype(int)
    return np.concatenate(out), y[idx], u[idx], lens


class BandNormalizer:
    """Standardisasi per pita (mean/std tiap baris frekuensi), di-fit pada TRAIN saja (anti leakage)."""

    def fit(self, X: np.ndarray):
        X = X.astype(np.float32)
        self.mean_ = X.mean(axis=(0, 2)).astype(np.float32)[:, None]
        self.std_ = (X.std(axis=(0, 2)) + 1e-6).astype(np.float32)[:, None]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X.astype(np.float32) - self.mean_) / self.std_).astype(np.float16)

    def state(self) -> dict:
        return {"mean": self.mean_, "std": self.std_}


class SpecDataset(Dataset):
    """Dataset spektrogram. Augmentasi (train=True) diterapkan on-the-fly per sampel:

    1. random crop  : potongan `crop` frame dengan offset acak di dalam area berisi suara
                      (rekaman lebih pendek dari crop → mulai dari 0, sisanya padding).
    2. random shift : geser melingkar ±`shift_max` frame di dalam potongan.
    3. SpecAugment  : `n_freq_masks` × lebar ≤ `freq_mask` pita dan `n_time_masks` × lebar ≤ `time_mask`
                      frame diisi 0 (= rata-rata setelah standardisasi).
    Mode eval: seluruh panjang (tanpa crop), tanpa augmentasi.
    """

    def __init__(self, X, y, lengths=None, train: bool = False, crop: int | None = 256,
                 shift_max: int = 20, freq_mask: int = 8, time_mask: int = 25, n_freq_masks: int = 2,
                 n_time_masks: int = 2, seed: int = 42):
        self.X, self.y = X, y
        self.lengths = lengths if lengths is not None else np.full(len(y), X.shape[2])
        self.train, self.crop, self.shift_max = train, crop, shift_max
        self.freq_mask, self.time_mask = freq_mask, time_mask
        self.n_freq_masks, self.n_time_masks = n_freq_masks, n_time_masks
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def _augment(self, x: np.ndarray, length: int) -> np.ndarray:
        r = self.rng
        T = x.shape[1]
        if self.crop and self.crop < T:
            hi = max(min(length, T) - self.crop, 0)
            s = int(r.integers(0, hi + 1))
            x = x[:, s:s + self.crop].copy()
        else:
            x = x.copy()
        if self.shift_max:
            x = np.roll(x, int(r.integers(-self.shift_max, self.shift_max + 1)), axis=1)
        Fb, T = x.shape
        for _ in range(self.n_freq_masks):
            w = int(r.integers(0, self.freq_mask + 1))
            f0 = int(r.integers(0, max(Fb - w, 0) + 1))
            x[f0:f0 + w, :] = 0
        for _ in range(self.n_time_masks):
            w = int(r.integers(0, self.time_mask + 1))
            t0 = int(r.integers(0, max(T - w, 0) + 1))
            x[:, t0:t0 + w] = 0
        return x

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32)
        if self.train:
            x = self._augment(x, int(self.lengths[i]))
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.tensor(self.y[i], dtype=torch.float32)


def mixup_batch(x, y, alpha: float = 0.2, gen: torch.Generator | None = None):
    """Mixup (Zhang dkk., 2018): x' = λx_i + (1−λ)x_j, y' = λy_i + (1−λ)y_j; λ ~ Beta(α, α)."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), generator=gen)
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _kernel(kernel: str):
    """'3x3' → (3, 3); '1x12' → (1, 12): receptive field sempit di frekuensi, lebar di waktu (R3)."""
    a, b = (int(v) for v in kernel.lower().split("x"))
    return (a, b)


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel=(3, 3), pool=(2, 2), dropout=0.1):
        super().__init__(
            nn.Conv2d(c_in, c_out, kernel, padding="same", bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool, ceil_mode=True),
            nn.Dropout(dropout),
        )


class LightCNN(nn.Module):
    """CNN ringan terinspirasi R3: blok [Conv2d → BatchNorm → ReLU → MaxPool → Dropout],
    filter 16-32-64-128(-128), Global Average Pooling, FC → 1 logit (BCEWithLogits).

    GAP membuat model menerima panjang waktu berapa pun (train: crop; eval: rekaman penuh).
    """

    def __init__(self, n_blocks: int = 4, filters=(16, 32, 64, 128, 128), kernel: str = "3x3",
                 dropout: float = 0.1, dropout_fc: float = 0.3):
        super().__init__()
        k = _kernel(kernel)
        chans = [1, *filters[:n_blocks]]
        self.features = nn.Sequential(*[
            ConvBlock(chans[i], chans[i + 1], k, (2, 2), dropout * (1 + i / max(n_blocks - 1, 1)))
            for i in range(n_blocks)
        ])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout_fc),
                                  nn.Linear(chans[-1], 1))
        self.config = dict(arch="LightCNN", n_blocks=n_blocks, filters=list(filters[:n_blocks]),
                           kernel=kernel, dropout=dropout, dropout_fc=dropout_fc)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class CRNN(nn.Module):
    """Pembanding kecil: 3 blok CNN (pool frekuensi lebih agresif) → urutan per frame → GRU dua arah
    → rata-rata waktu → FC(1)."""

    def __init__(self, n_freq: int, filters=(16, 32, 64), hidden: int = 64, dropout: float = 0.1,
                 dropout_fc: float = 0.3):
        super().__init__()
        chans = [1, *filters]
        self.features = nn.Sequential(*[
            ConvBlock(chans[i], chans[i + 1], (3, 3), (2, 2) if i == 0 else (2, 1) if i == 1 else (2, 2),
                      dropout) for i in range(len(filters))
        ])
        f_out = math.ceil(n_freq / 8)
        self.gru = nn.GRU(filters[-1] * f_out, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout_fc), nn.Linear(2 * hidden, 1))
        self.config = dict(arch="CRNN", n_freq=n_freq, filters=list(filters), hidden=hidden,
                           dropout=dropout, dropout_fc=dropout_fc)

    def forward(self, x):
        z = self.features(x)                              # (B, C, F', T')
        z = z.permute(0, 3, 1, 2).flatten(2)              # (B, T', C·F')
        z, _ = self.gru(z)
        return self.head(z.mean(1)).squeeze(1)


def build_model(config: dict) -> nn.Module:
    cfg = dict(config)
    arch = cfg.pop("arch", "LightCNN")
    if arch == "CRNN":
        return CRNN(**cfg)
    return LightCNN(**cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_table(model: nn.Module, input_shape=(1, 1, 64, 256)) -> pd.DataFrame:
    """Tabel layer: nama, tipe, bentuk keluaran, jumlah parameter (hook pada modul daun)."""
    rows, hooks = [], []

    def hook(mod, inp, out, name):
        o = out[0] if isinstance(out, tuple) else out
        rows.append({"layer": name, "tipe": type(mod).__name__, "output": tuple(o.shape[1:]),
                     "parameter": sum(p.numel() for p in mod.parameters(recurse=False))})

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: hook(mod, i, o, n)))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for h in hooks:
        h.remove()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inferensi & interpretasi
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(xb.to(device))).cpu().numpy() for xb, _ in loader]
    return np.concatenate(out)


def grad_cam(model: LightCNN, x: torch.Tensor, layer: nn.Module | None = None) -> tuple[np.ndarray, float]:
    """Grad-CAM (Selvaraju dkk., 2017) terhadap logit COVID-19.

    Bobot kanal = rata-rata gradien logit terhadap aktivasi layer target; peta = ReLU(Σ bobot·aktivasi),
    diinterpolasi ke ukuran input dan dinormalisasi 0..1. x: (1, 1, F, T).
    """
    layer = layer or model.features[-1][2]   # ReLU blok terakhir (sebelum pooling)
    acts, grads = {}, {}

    def fwd_hook(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))

    h1 = layer.register_forward_hook(fwd_hook)
    model.eval()
    x = x.clone().requires_grad_(True)
    logit = model(x)
    model.zero_grad()
    logit.sum().backward()
    h1.remove()
    a, g = acts["a"].detach(), grads["g"].detach()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0].numpy()
    return cam / (cam.max() + 1e-8), float(torch.sigmoid(logit).item())
