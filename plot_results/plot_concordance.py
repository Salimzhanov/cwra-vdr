#!/usr/bin/env python3
"""Plot pairwise modality concordance as a lower-triangle heatmap.

Default mode uses all 11 modalities (including Boltz-2 confidence).
Use `--mode paper10` to reproduce the 10-modality reference layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

Modality = Tuple[str, str, str]  # (column, display_label, direction)


PAPER10_MODALITIES: List[Modality] = [
    ("graphdta_kd", "GraphDTA\n$K_d$", "high"),
    ("graphdta_ki", "GraphDTA\n$K_i$", "high"),
    ("graphdta_ic50", "GraphDTA\n$IC_{50}$", "high"),
    ("mltle_pKd", "MLT-LE\n$pKd$", "high"),
    ("vina_score", "Vina\nScore", "low"),
    ("boltz_affinity", "Boltz-2\nAffinity", "low"),
    ("unimol_similarity", "UniMol\nSimilarity", "high"),
    ("tankbind_affinity", "TANKBind\nAffinity", "high"),
    ("drugban_affinity", "DrugBAN\nAffinity", "low"),
    ("moltrans_affinity", "MolTrans\nAffinity", "low"),
]


ALL11_MODALITIES: List[Modality] = [
    ("graphdta_kd", "GraphDTA\n$K_d$", "high"),
    ("graphdta_ki", "GraphDTA\n$K_i$", "high"),
    ("graphdta_ic50", "GraphDTA\n$IC_{50}$", "high"),
    ("mltle_pKd", "MLT-LE\n$pKd$", "high"),
    ("tankbind_affinity", "TANKBind\nAffinity", "high"),
    ("drugban_affinity", "DrugBAN\nAffinity", "low"),
    ("moltrans_affinity", "MolTrans\nAffinity", "low"),
    ("vina_score", "Vina\nScore", "low"),
    ("boltz_affinity", "Boltz-2\nAffinity", "low"),
    ("boltz_confidence", "Boltz-2\nConfidence", "low"),
    ("unimol_similarity", "UniMol\nSimilarity", "high"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/composed_modalities_with_rdkit.csv",
        help="Input CSV file with modality columns.",
    )
    parser.add_argument(
        "--mode",
        choices=["paper10", "all11"],
        default="all11",
        help="`all11` uses all modalities (default); `paper10` matches the uploaded 10-modality layout.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path (default: plor_results/concordance_heatmap10.pdf or ...11.pdf).",
    )
    parser.add_argument(
        "--matrix-out",
        default=None,
        help="Optional CSV path to save the full concordance matrix.",
    )
    parser.add_argument("--vmin", type=float, default=0.30, help="Heatmap lower color bound.")
    parser.add_argument("--vmax", type=float, default=0.75, help="Heatmap upper color bound.")
    parser.add_argument(
        "--light-text-threshold",
        type=float,
        default=0.62,
        help="Annotation text switches to light color above this value.",
    )
    return parser.parse_args()


def get_modalities(mode: str) -> Sequence[Modality]:
    if mode == "paper10":
        return PAPER10_MODALITIES
    return ALL11_MODALITIES


def _as_signed_numeric(series: pd.Series, direction: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if direction == "low":
        values = -values
    return values


class _FenwickTree:
    """Fenwick tree for prefix sums on non-negative integer indices."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int, value: int = 1) -> None:
        i = index + 1
        while i <= self.size:
            self.tree[i] += value
            i += i & -i

    def prefix_sum(self, index: int) -> int:
        i = index + 1
        total = 0
        while i > 0:
            total += int(self.tree[i])
            i -= i & -i
        return total


def _choose2(counts: np.ndarray) -> int:
    counts = counts.astype(np.int64, copy=False)
    return int(np.sum(counts * (counts - 1) // 2))


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Kendall's tau-b without SciPy (O(n log n))."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = x.size
    if n < 2:
        return np.nan

    order = np.lexsort((y, x))
    x = x[order]
    y = y[order]

    n0 = n * (n - 1) // 2
    _, x_counts = np.unique(x, return_counts=True)
    _, y_counts = np.unique(y, return_counts=True)
    xtie = _choose2(x_counts)
    ytie = _choose2(y_counts)

    xy = np.column_stack((x, y))
    _, xy_counts = np.unique(xy, axis=0, return_counts=True)
    ntie = _choose2(xy_counts)

    _, y_rank = np.unique(y, return_inverse=True)
    bit = _FenwickTree(int(y_rank.max()) + 1)
    discordant = 0
    processed = 0

    i = 0
    while i < n:
        j = i + 1
        while j < n and x[j] == x[i]:
            j += 1

        group = y_rank[i:j]
        for rank in group:
            discordant += processed - bit.prefix_sum(int(rank))

        for rank in group:
            bit.add(int(rank), 1)

        processed += j - i
        i = j

    concordant_minus_discordant = n0 - xtie - ytie + ntie - 2 * discordant
    denom = np.sqrt((n0 - xtie) * (n0 - ytie))
    if denom == 0:
        return np.nan
    return float(concordant_minus_discordant / denom)


def compute_concordance_matrix(df: pd.DataFrame, modalities: Sequence[Modality]) -> pd.DataFrame:
    cols = [m[0] for m in modalities]
    labels = [m[1] for m in modalities]
    directions = {m[0]: m[2] for m in modalities}

    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required modality columns: {missing}")

    signed_arrays = {
        col: _as_signed_numeric(df[col], directions[col]).to_numpy(dtype=float)
        for col in cols
    }

    matrix = np.eye(len(cols), dtype=float)

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if i <= j:
                continue
            tau = kendall_tau_b(signed_arrays[col_i], signed_arrays[col_j])
            concordance = np.nan if np.isnan(tau) else 0.5 * (tau + 1.0)

            matrix[i, j] = concordance
            matrix[j, i] = concordance

    return pd.DataFrame(matrix, index=labels, columns=labels)


def plot_concordance(
    matrix: pd.DataFrame,
    output_path: Path,
    vmin: float,
    vmax: float,
    light_text_threshold: float,
) -> None:
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        pass

    bg = "#EAEAF2"
    fig, ax = plt.subplots(figsize=(8.8, 8.1))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.grid(False)

    data = matrix.to_numpy(dtype=float)
    n = data.shape[0]
    lower = np.full_like(data, np.nan, dtype=float)

    for i in range(n):
        for j in range(n):
            if i > j:
                lower[i, j] = data[i, j]

    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color=bg)
    image = ax.imshow(lower, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    for i in range(n):
        for j in range(n):
            if i > j and np.isfinite(lower[i, j]):
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="white",
                        linewidth=1.0,
                    )
                )
                value = lower[i, j]
                text_color = "#d8e3f2" if value >= light_text_threshold else "#1a1a1a"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=text_color,
                )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="x", labelrotation=45, labelsize=12)
    ax.tick_params(axis="y", labelrotation=0, labelsize=12)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(list(matrix.columns))
    ax.set_yticklabels(list(matrix.index))
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
        tick.set_rotation_mode("anchor")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    ticks = np.round(np.arange(vmin, vmax + 1e-9, 0.05), 2)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel("Concordance Index", fontsize=15, rotation=90, labelpad=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    modalities = get_modalities(args.mode)
    df = pd.read_csv(args.input)

    matrix = compute_concordance_matrix(df, modalities)

    default_name = "concordance_heatmap10.pdf" if args.mode == "paper10" else "concordance_heatmap11.pdf"
    output_path = Path(args.output) if args.output else Path("plor_results") / default_name
    plot_concordance(
        matrix=matrix,
        output_path=output_path,
        vmin=args.vmin,
        vmax=args.vmax,
        light_text_threshold=args.light_text_threshold,
    )

    if args.matrix_out:
        matrix_out = Path(args.matrix_out)
    else:
        stem = output_path.stem
        matrix_out = output_path.with_name(f"{stem}_matrix.csv")
    matrix.to_csv(matrix_out, index=True)

    print(f"Saved figure: {output_path}")
    print(f"Saved matrix: {matrix_out}")


if __name__ == "__main__":
    main()
