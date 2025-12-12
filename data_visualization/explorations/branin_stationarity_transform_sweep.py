"""
Explorations: Branin output transforms for improved stationarity.

We compare monotone variance-compressing transforms and quantify how uniformly the
local gradient magnitude varies across the domain. A more uniform gradient map is a
proxy for "more stationary" behavior under a stationary kernel with a single global
lengthscale.

Run:
  python data_visualization/explorations/branin_stationarity_transform_sweep.py

Outputs:
  - exp_branin_transform_stationarity_grid.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import matplotlib.pyplot as plt


def branin(
    x1: np.ndarray,
    x2: np.ndarray,
    a: float = 1.0,
    b: float = 5.1 / (4 * np.pi**2),
    c: float = 5.0 / np.pi,
    r: float = 6.0,
    s: float = 10.0,
    t: float = 1.0 / (8 * np.pi),
) -> np.ndarray:
    term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)
    return term1 + term2 + s


def gradient_magnitude(F: np.ndarray, dx1: float, dx2: float) -> np.ndarray:
    grad_x1 = np.gradient(F, dx1, axis=1)
    grad_x2 = np.gradient(F, dx2, axis=0)
    return np.sqrt(grad_x1**2 + grad_x2**2)


@dataclass(frozen=True)
class GradMetrics:
    p5: float
    p50: float
    p95: float
    ratio_95_5: float
    ratio_99_1: float
    cv: float


def compute_grad_metrics(g: np.ndarray) -> GradMetrics:
    g_flat = g.ravel()
    p1, p5, p50, p95, p99 = np.percentile(g_flat, [1, 5, 50, 95, 99])
    eps = 1e-12
    ratio_95_5 = float(p95 / (p5 + eps))
    ratio_99_1 = float(p99 / (p1 + eps))
    cv = float(g_flat.std() / (g_flat.mean() + eps))
    return GradMetrics(
        p5=float(p5),
        p50=float(p50),
        p95=float(p95),
        ratio_95_5=ratio_95_5,
        ratio_99_1=ratio_99_1,
        cv=cv,
    )


def main() -> None:
    save_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(save_dir, "exp_branin_transform_stationarity_grid.png")

    bounds = [(-5.0, 10.0), (0.0, 15.0)]
    n = 500
    x1 = np.linspace(bounds[0][0], bounds[0][1], n)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n)
    X1, X2 = np.meshgrid(x1, x2)
    Z = branin(X1, X2)

    dx1 = float(x1[1] - x1[0])
    dx2 = float(x2[1] - x2[0])

    transforms: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "identity": lambda y: y,
        "log(y+1)": lambda y: np.log(y + 1.0),
        "sqrt(y)": lambda y: np.sqrt(y),
    }

    results: List[dict] = []
    for name, fn in transforms.items():
        F = fn(Z)
        g = gradient_magnitude(F, dx1=dx1, dx2=dx2)
        metrics = compute_grad_metrics(g)
        g_rel = g / (np.median(g.ravel()) + 1e-12)
        results.append(
            {
                "name": name,
                "F": F,
                "g_rel": g_rel,
                "metrics": metrics,
            }
        )

    # Shared color limits for relative gradient variation (log10 scale)
    all_log_g = np.concatenate([np.log10(r["g_rel"].ravel() + 1e-12) for r in results])
    vmin, vmax = np.percentile(all_log_g, [1, 99])

    fig, axes = plt.subplots(2, len(results), figsize=(15, 7), constrained_layout=True)
    if len(results) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    minima = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]

    for col, r in enumerate(results):
        name = r["name"]
        F = r["F"]
        g_rel = r["g_rel"]
        m: GradMetrics = r["metrics"]

        ax_f = axes[0, col]
        im_f = ax_f.imshow(
            F,
            extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        for x, y in minima:
            ax_f.plot(x, y, "r*", markersize=9, markeredgecolor="white", markeredgewidth=1.0)
        ax_f.set_title(name)
        ax_f.set_xlabel("$x_1$")
        ax_f.set_ylabel("$x_2$")
        plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.02)

        ax_g = axes[1, col]
        im_g = ax_g.imshow(
            np.log10(g_rel + 1e-12),
            extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        for x, y in minima:
            ax_g.plot(x, y, "k*", markersize=7, markeredgecolor="white", markeredgewidth=0.8)
        ax_g.set_title(f"log10(|∇F| / median)\n(p95/p5={m.ratio_95_5:.2f}, CV={m.cv:.2f})")
        ax_g.set_xlabel("$x_1$")
        ax_g.set_ylabel("$x_2$")
        plt.colorbar(im_g, ax=ax_g, fraction=0.046, pad=0.02, label="log10(rel. grad)")

    fig.suptitle("Branin: output transforms and relative gradient variation", fontsize=13)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Branin stationarity proxy: gradient magnitude variation (lower is better)")
    for r in results:
        m: GradMetrics = r["metrics"]
        print(
            f"  {r['name']:<10s}  p95/p5={m.ratio_95_5:6.2f}  p99/p1={m.ratio_99_1:6.2f}  CV={m.cv:5.2f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

