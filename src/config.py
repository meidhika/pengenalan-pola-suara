"""Loader konfigurasi proyek (config.yaml) dan utilitas seed."""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path

import yaml

# Root proyek = folder induk dari src/
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


class _Cfg(dict):
    """dict yang juga bisa diakses dengan atribut (cfg.audio.sample_rate)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e


def _wrap(obj):
    if isinstance(obj, dict):
        return _Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> _Cfg:
    """Baca config.yaml; semua entri `paths` diubah menjadi Path absolut."""
    path = Path(path) if path else CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: (ROOT / v).resolve() for k, v in cfg["paths"].items()}
    return _wrap(cfg)


def set_seed(seed: int | None = None) -> int:
    """Set seed untuk random, numpy, dan torch (bila terpasang)."""
    seed = load_config().seed if seed is None else seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


CFG = load_config()
