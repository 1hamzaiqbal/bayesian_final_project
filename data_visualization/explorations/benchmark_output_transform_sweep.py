"""
Explorations: output transforms for LDA and SVM benchmark objectives.

We compare simple transforms (log / log1p) against a learned power transform (Box-Cox).
To make shapes comparable across transforms, we z-score each transformed distribution
and overlay KDEs against N(0, 1).

Run:
  python data_visualization/explorations/benchmark_output_transform_sweep.py

Outputs:
  - exp_benchmark_output_transform_zscore_kde.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PowerTransformer


@dataclass(frozen=True)
class DistSummary:
    skew: float
    kurtosis_fisher: float


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return (x - x.mean()) / (x.std() + 1e-12)


def summarize(x: np.ndarray) -> DistSummary:
    return DistSummary(
        skew=float(stats.skew(x)),
        kurtosis_fisher=float(stats.kurtosis(x)),
    )


def kde_curve(x: np.ndarray, xs: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    x = x[np.abs(x) < 10]  # avoid extreme tails dominating the plot
    if len(x) < 2:
        return np.zeros_like(xs)
    return stats.gaussian_kde(x)(xs)


def boxcox_transform(y: np.ndarray) -> Tuple[np.ndarray, float]:
    pt = PowerTransformer(method="box-cox", standardize=False)
    y_t = pt.fit_transform(y.reshape(-1, 1)).ravel()
    return y_t, float(pt.lambdas_[0])


def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(save_dir, "exp_benchmark_output_transform_zscore_kde.png")

    lda = np.loadtxt(os.path.join(project_root, "lda.csv"), delimiter=",")
    svm = np.loadtxt(os.path.join(project_root, "svm.csv"), delimiter=",")

    lda_y = lda[:, 3]
    svm_y = svm[:, 3]

    lda_boxcox, lda_lam = boxcox_transform(lda_y)
    svm_boxcox, svm_lam = boxcox_transform(svm_y)

    transforms: Dict[str, Dict[str, np.ndarray]] = {
        "LDA": {
            "identity": lda_y,
            "log(y)": np.log(lda_y),
            "Box-Cox(λ*)": lda_boxcox,
        },
        "SVM": {
            "identity": svm_y,
            "log(y)": np.log(svm_y),
            "log(y+1)": np.log1p(svm_y),
            "Box-Cox(λ*)": svm_boxcox,
        },
    }

    xs = np.linspace(-5, 5, 400)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, (dataset_name, tmap) in zip(axes, transforms.items()):
        ax.plot(xs, stats.norm.pdf(xs), "k--", label="N(0,1)")
        for tname, arr in tmap.items():
            z = zscore(arr)
            s = summarize(z)
            ax.plot(xs, kde_curve(z, xs), label=f"{tname} (skew={s.skew:+.2f})")
        ax.set_title(f"{dataset_name}: z-scored output distributions")
        ax.set_xlabel("z-score")
        ax.set_ylabel("density")
        ax.set_xlim(-5, 5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Benchmark outputs: comparing monotone transforms", fontsize=13)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Output-transform shape summaries (after z-scoring; lower |skew| is 'more symmetric'):")
    for dataset_name, tmap in transforms.items():
        print(f"\n{dataset_name}")
        for tname, arr in tmap.items():
            z = zscore(arr)
            s = summarize(z)
            print(f"  {tname:<10s}  skew={s.skew:+.3f}  kurt={s.kurtosis_fisher:+.3f}")

    print(f"\nBox-Cox lambdas: LDA λ={lda_lam:.3f}, SVM λ={svm_lam:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
