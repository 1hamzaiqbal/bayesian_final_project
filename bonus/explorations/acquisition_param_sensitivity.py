"""
Explorations: sensitivity of acquisition parameters under improved modeling.

Focus:
  - SVM: LCB κ sweep (LCB performed best in the bonus study)
  - LDA: PI ξ sweep (PI often strong)

Modeling choices (aligned with improved modeling sweep):
  - Branin: (not evaluated here)
  - LDA/SVM: log(y+1) + Matern 3/2 + corrected noise scaling

Outputs:
  - svm_kappa_sensitivity_improved.png
  - lda_pi_xi_sensitivity_improved.png
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BONUS_DIR = os.path.join(PROJECT_ROOT, "bonus")
sys.path.insert(0, BONUS_DIR)
import bonus_acquisition_comparison as base  # noqa: E402


def fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    kernel,
    noise_level: float = 0.001,
    n_restarts: int = 6,
) -> GaussianProcessRegressor:
    # Correct noise scaling under normalize_y=True so sigma is in original units
    alpha = noise_level**2
    y_std = float(np.std(y))
    if y_std > 0:
        alpha = (noise_level / y_std) ** 2

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=42,
    )
    gp.fit(X, y)
    return gp


def run_bo(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    init_indices: np.ndarray,
    acquisition: str,
    acq_params: Dict[str, float],
    kernel,
    use_log_transform: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    available = np.ones(len(X_pool), dtype=bool)
    selected: List[int] = []
    y_values: List[float] = []
    best_so_far: List[float] = []

    for idx in init_indices:
        idx_int = int(idx)
        selected.append(idx_int)
        y_values.append(float(y_pool[idx_int]))
        available[idx_int] = False
        best_so_far.append(float(np.min(y_values)))

    for _ in range(30):
        X_train = X_pool[selected]
        y_train = np.array(y_values)

        if use_log_transform:
            y_fit = np.log(y_train + 1)
        else:
            y_fit = y_train

        gp = fit_gp(X_train, y_fit, kernel)

        avail_idx = np.where(available)[0]
        mu, sigma = gp.predict(X_pool[avail_idx], return_std=True)

        if use_log_transform:
            f_best = float(np.log(np.min(y_values) + 1))
        else:
            f_best = float(np.min(y_values))

        if acquisition == "LCB":
            acq = base.lower_confidence_bound(mu, sigma, kappa=acq_params["kappa"])
            next_global = int(avail_idx[int(np.argmax(acq))])
        elif acquisition == "PI":
            acq = base.probability_of_improvement(mu, sigma, f_best, xi=acq_params["xi"])
            next_global = int(avail_idx[int(np.argmax(acq))])
        else:
            raise ValueError(f"Unsupported acquisition: {acquisition}")

        selected.append(next_global)
        y_values.append(float(y_pool[next_global]))
        available[next_global] = False
        best_so_far.append(float(np.min(y_values)))

    return np.array(best_so_far)


def gap_from_best(best_so_far: np.ndarray, f_opt: float) -> float:
    f_best_initial = float(best_so_far[4])
    f_best_found = float(best_so_far[-1])
    return float(base.compute_gap(f_best_found, f_best_initial, f_opt))


def load_pools() -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
    lda_data = np.loadtxt(os.path.join(PROJECT_ROOT, "lda.csv"), delimiter=",")
    svm_data = np.loadtxt(os.path.join(PROJECT_ROOT, "svm.csv"), delimiter=",")
    return {
        "LDA": (lda_data[:, :3], lda_data[:, 3], float(lda_data[:, 3].min())),
        "SVM": (svm_data[:, :3], svm_data[:, 3], float(svm_data[:, 3].min())),
    }


@dataclass(frozen=True)
class SweepResult:
    xs: List[float]
    means: List[float]
    ses: List[float]
    per_value: Dict[float, np.ndarray]


def sweep_param(
    dataset_name: str,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    f_opt: float,
    acquisition: str,
    param_name: str,
    values: List[float],
    n_runs: int = 20,
) -> SweepResult:
    n_pool = len(X_pool)
    all_init = []
    for run in range(n_runs):
        run_rng = np.random.default_rng(seed=run * 42 + 1)
        all_init.append(run_rng.choice(n_pool, size=5, replace=False))

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0, 1.0, 1.0],
        length_scale_bounds=(1e-2, 1e2),
        nu=1.5,
    )

    per_value: Dict[float, np.ndarray] = {}
    for v in values:
        gaps = []
        for run in range(n_runs):
            init_indices = all_init[run].copy()
            rng = np.random.default_rng(seed=run * 999 + 5)
            best = run_bo(
                X_pool,
                y_pool,
                init_indices,
                acquisition=acquisition,
                acq_params={param_name: v},
                kernel=kernel,
                use_log_transform=True,
                rng=rng,
            )
            gaps.append(gap_from_best(best, f_opt))
        per_value[v] = np.array(gaps)

    xs = values
    means = [float(per_value[v].mean()) for v in values]
    ses = [float(per_value[v].std() / np.sqrt(n_runs)) for v in values]
    return SweepResult(xs=xs, means=means, ses=ses, per_value=per_value)


def annotate_best(ax, result: SweepResult, label: str):
    best_idx = int(np.argmax(result.means))
    best_x = result.xs[best_idx]
    best_mean = result.means[best_idx]

    # Compare top-2 via paired t-test
    sorted_idx = np.argsort(result.means)[::-1]
    annotation = f"Best {label}={best_x:g}"
    if len(sorted_idx) >= 2:
        a = result.per_value[result.xs[int(sorted_idx[0])]]
        b = result.per_value[result.xs[int(sorted_idx[1])]]
        _, p = stats.ttest_rel(a, b)
        if p > 0.05:
            annotation = f"Top two indistinguishable (p={p:.2f})"

    ax.annotate(
        annotation,
        xy=(best_x, best_mean),
        xytext=(best_x, best_mean - 0.08),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
        ha="center",
    )


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    pools = load_pools()

    # SVM: κ sweep
    X_svm, y_svm, f_opt_svm = pools["SVM"]
    kappa_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    svm_res = sweep_param(
        "SVM",
        X_svm,
        y_svm,
        f_opt_svm,
        acquisition="LCB",
        param_name="kappa",
        values=kappa_values,
        n_runs=20,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(svm_res.xs, svm_res.means, yerr=svm_res.ses, marker="o", capsize=4, linewidth=2)
    ax.set_xlabel("κ (LCB exploration weight)")
    ax.set_ylabel("Mean final gap @35 (±SE)")
    ax.set_title("SVM: LCB κ sensitivity (Matern 3/2, log(y+1), corrected noise)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    annotate_best(ax, svm_res, label="κ")
    plt.tight_layout()
    out = os.path.join(save_dir, "svm_kappa_sensitivity_improved.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    print("\nSVM κ sweep (mean ±SE):")
    for kappa, mean, se in zip(svm_res.xs, svm_res.means, svm_res.ses):
        print(f"  κ={kappa:<4g}  {mean:.4f} ± {se:.4f}")

    # LDA: ξ sweep for PI
    X_lda, y_lda, f_opt_lda = pools["LDA"]
    xi_values = [0.0, 0.001, 0.01, 0.05, 0.1]
    lda_res = sweep_param(
        "LDA",
        X_lda,
        y_lda,
        f_opt_lda,
        acquisition="PI",
        param_name="xi",
        values=xi_values,
        n_runs=20,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(lda_res.xs, lda_res.means, yerr=lda_res.ses, marker="o", capsize=4, linewidth=2)
    ax.set_xlabel("ξ (PI improvement margin)")
    ax.set_ylabel("Mean final gap @35 (±SE)")
    ax.set_title("LDA: PI ξ sensitivity (Matern 3/2, log(y+1), corrected noise)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    annotate_best(ax, lda_res, label="ξ")
    plt.tight_layout()
    out = os.path.join(save_dir, "lda_pi_xi_sensitivity_improved.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    print("\nLDA PI ξ sweep (mean ±SE):")
    for xi, mean, se in zip(lda_res.xs, lda_res.means, lda_res.ses):
        print(f"  ξ={xi:<5g}  {mean:.4f} ± {se:.4f}")


if __name__ == "__main__":
    main()
