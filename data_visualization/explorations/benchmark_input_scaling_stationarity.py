"""
Explorations: input scaling for LDA/SVM hyperparameter grids.

The benchmark hyperparameter grids span orders of magnitude in some dimensions (especially
SVM). A GP on raw inputs can therefore look non-stationary simply due to coordinate scaling.
A common fix is to model in log-space for positive hyperparameters.

This script measures a simple proxy: the distribution of |∇f| over the full 3D grid under
different coordinate systems (raw vs log-transformed inputs).

Run:
  python data_visualization/explorations/benchmark_input_scaling_stationarity.py

Outputs:
  - exp_benchmark_input_scaling_gradmag_kde.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


@dataclass(frozen=True)
class GradSummary:
    p5: float
    p50: float
    p95: float
    ratio_95_5: float
    cv: float


def build_grid(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1_vals = np.unique(data[:, 0])
    x2_vals = np.unique(data[:, 1])
    x3_vals = np.unique(data[:, 2])

    x1_vals.sort()
    x2_vals.sort()
    x3_vals.sort()

    i1 = {float(v): i for i, v in enumerate(x1_vals)}
    i2 = {float(v): i for i, v in enumerate(x2_vals)}
    i3 = {float(v): i for i, v in enumerate(x3_vals)}

    Y = np.full((len(x1_vals), len(x2_vals), len(x3_vals)), np.nan, dtype=float)
    for row in data:
        Y[i1[float(row[0])], i2[float(row[1])], i3[float(row[2])]] = float(row[3])

    if not np.all(np.isfinite(Y)):
        raise ValueError("Grid assembly failed: missing values in Y.")

    return x1_vals, x2_vals, x3_vals, Y


def grad_mag(Y: np.ndarray, coords: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    g1, g2, g3 = np.gradient(Y, coords[0], coords[1], coords[2], edge_order=1)
    return np.sqrt(g1**2 + g2**2 + g3**2)


def summarize_grad(g: np.ndarray) -> GradSummary:
    g_flat = g.ravel()
    g_flat = g_flat[np.isfinite(g_flat)]
    eps = 1e-12
    p5, p50, p95 = np.percentile(g_flat, [5, 50, 95])
    ratio_95_5 = float(p95 / (p5 + eps))
    cv = float(g_flat.std() / (g_flat.mean() + eps))
    return GradSummary(
        p5=float(p5),
        p50=float(p50),
        p95=float(p95),
        ratio_95_5=ratio_95_5,
        cv=cv,
    )


def kde_curve(x: np.ndarray, xs: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    x = x[np.abs(x) < 50]
    if len(x) < 2:
        return np.zeros_like(xs)
    return stats.gaussian_kde(x)(xs)


def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(save_dir, "exp_benchmark_input_scaling_gradmag_kde.png")

    lda = np.loadtxt(os.path.join(project_root, "lda.csv"), delimiter=",")
    svm = np.loadtxt(os.path.join(project_root, "svm.csv"), delimiter=",")

    datasets: Dict[str, np.ndarray] = {
        "LDA": lda,
        "SVM": svm,
    }

    xs = np.linspace(-10, 10, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, (name, data) in zip(axes, datasets.items()):
        x1, x2, x3, Y = build_grid(data)

        configs: List[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
            ("raw inputs", (x1, x2, x3)),
        ]

        if name == "LDA":
            configs.append(("log10 dims 2&3", (x1, np.log10(x2), np.log10(x3))))
        else:
            configs.append(("log10 dims 1&3", (np.log10(x1), x2, np.log10(x3))))

        summaries: Dict[str, GradSummary] = {}
        for label, coords in configs:
            g = grad_mag(Y, coords=coords)
            summaries[label] = summarize_grad(g)
            g_plot = np.log10(g.ravel() + 1e-12)
            ax.plot(xs, kde_curve(g_plot, xs), label=f"{label} (p95/p5={summaries[label].ratio_95_5:.1f})")

        ax.set_title(f"{name}: log10(|∇f|) under input scalings")
        ax.set_xlabel("log10(|∇f|)")
        ax.set_ylabel("density")
        ax.set_xlim(xs.min(), xs.max())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        print(f"\n{name} input-scaling stationarity proxy (lower is better):")
        for label, s in summaries.items():
            print(f"  {label:<16s}  p95/p5={s.ratio_95_5:6.2f}  CV={s.cv:5.2f}  p50={s.p50:.4g}")

    fig.suptitle("Input scaling affects apparent stationarity on benchmark grids", fontsize=13)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

