"""Tahap 7a — membuat diagram alur penelitian (reports/figures/07_alur_penelitian.png).

Pemakaian:  python scripts/07_diagram_alur.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plots import plot_research_flow, save_fig  # noqa: E402

if __name__ == "__main__":
    out = save_fig(plot_research_flow(), "07_alur_penelitian")
    print(f"Tersimpan: {out}")
