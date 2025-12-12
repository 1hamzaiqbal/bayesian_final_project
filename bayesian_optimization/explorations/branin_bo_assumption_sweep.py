"""
Explorations: sweep key Bayesian optimization assumptions on Branin.

We compare:
  A) Baseline (log transform + SE kernel, old noise scaling)
  B) Baseline + corrected noise scaling (sigma in original units)
  C) Improved surrogate: original scale + SE + Periodic(x1) kernel + corrected noise

Outputs:
  - branin_bo_learning_comparison.png
  - printed table of mean final gap and best values
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel,
    ConstantKernel,
    RBF,
    ExpSineSquared,
)

# Import baseline EI and Branin definition from main module
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
import bayesian_optimization as bo  # noqa: E402


class ActiveDimKernel(Kernel):
    """Apply a base kernel to selected input dimensions."""

    def __init__(self, base_kernel, active_dims):
        self.base_kernel = base_kernel
        self.active_dims = tuple(active_dims)

    def __call__(self, X, Y=None, eval_gradient=False):
        X_sub = X[:, self.active_dims]
        Y_sub = None if Y is None else Y[:, self.active_dims]
        return self.base_kernel(X_sub, Y_sub, eval_gradient=eval_gradient)

    def diag(self, X):
        return self.base_kernel.diag(X[:, self.active_dims])

    def is_stationary(self):
        return self.base_kernel.is_stationary()

    @property
    def hyperparameters(self):
        return self.base_kernel.hyperparameters

    @property
    def theta(self):
        return self.base_kernel.theta

    @theta.setter
    def theta(self, theta):
        self.base_kernel.theta = theta

    @property
    def bounds(self):
        return self.base_kernel.bounds

    def get_params(self, deep=True):
        return {"base_kernel": self.base_kernel, "active_dims": self.active_dims}

    def clone_with_theta(self, theta):
        return ActiveDimKernel(self.base_kernel.clone_with_theta(theta), self.active_dims)


def fit_gp_variant(X, y, kernel, noise_level=0.001, normalize_y=True, corrected_noise=False, n_restarts=10):
    """
    Fit GP with either baseline noise scaling or corrected scaling.

    If corrected_noise=True and normalize_y=True, alpha is scaled by y_std so that
    noise_level is interpreted in original output units.
    """
    alpha = noise_level ** 2
    if corrected_noise and normalize_y:
        y_std = np.std(y)
        if y_std > 0:
            alpha = (noise_level / y_std) ** 2

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts,
        random_state=42,
    )
    gp.fit(X, y)
    return gp


def bo_run(X_pool, y_pool, init_indices, kernel, use_log, corrected_noise):
    """Run a single BO trajectory for Branin with given choices."""
    available = np.ones(len(X_pool), dtype=bool)
    selected = []
    y_obs = []
    best_so_far = []

    for idx in init_indices:
        selected.append(idx)
        y_obs.append(y_pool[idx])
        available[idx] = False
        best_so_far.append(min(y_obs))

    for _ in range(30):
        X_train = X_pool[selected]
        y_train = np.array(y_obs)
        if use_log:
            y_fit = np.log(y_train + 1)
        else:
            y_fit = y_train

        gp = fit_gp_variant(
            X_train,
            y_fit,
            kernel=kernel,
            noise_level=0.001,
            normalize_y=True,
            corrected_noise=corrected_noise,
            n_restarts=12,
        )

        avail_idx = np.where(available)[0]
        mu, sigma = gp.predict(X_pool[avail_idx], return_std=True)

        if use_log:
            f_best = np.log(min(y_obs) + 1)
        else:
            f_best = min(y_obs)

        ei = bo.expected_improvement(mu, sigma, f_best, xi=0.01)
        next_local = np.argmax(ei)
        next_global = avail_idx[next_local]

        selected.append(next_global)
        y_obs.append(y_pool[next_global])
        available[next_global] = False
        best_so_far.append(min(y_obs))

    return np.array(best_so_far)


def compute_gap_curve(best_so_far, f_opt):
    f_best_initial = best_so_far[4]
    gaps = []
    for best in best_so_far:
        gaps.append(bo.compute_gap(best, f_best_initial, f_opt))
    return np.array(gaps)


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    bounds = [(-5, 10), (0, 15)]

    # Candidate pool (same as main)
    sampler = qmc.Sobol(d=2, scramble=True, seed=123)
    X_pool = sampler.random(1000)
    X_pool = qmc.scale(X_pool, [-5, 0], [10, 15])
    y_pool = bo.branin(X_pool[:, 0], X_pool[:, 1])
    f_opt = float(y_pool.min())

    # Kernels
    rbf = RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
    baseline_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * rbf

    periodic_x1 = ActiveDimKernel(
        ExpSineSquared(length_scale=1.0, periodicity=2 * np.pi),
        active_dims=[0],
    )
    improved_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * (rbf + periodic_x1)

    variants = [
        ("Baseline log + SE", baseline_kernel, True, False),
        ("Log + SE (corrected noise)", baseline_kernel, True, True),
        ("Orig + SE+Periodic(x1)", improved_kernel, False, True),
    ]

    n_runs = 20
    curves = {name: [] for name, *_ in variants}

    for run in range(n_runs):
        rng = np.random.default_rng(seed=run * 101 + 7)
        init_indices = rng.choice(len(X_pool), size=5, replace=False)

        for name, kernel, use_log, corrected_noise in variants:
            best = bo_run(X_pool, y_pool, init_indices, kernel, use_log, corrected_noise)
            gap_curve = compute_gap_curve(best, f_opt)
            curves[name].append(gap_curve)

    # Aggregate
    summary = []
    for name in curves:
        arr = np.vstack(curves[name])
        mean_curve = arr.mean(axis=0)
        se_curve = arr.std(axis=0) / np.sqrt(n_runs)
        summary.append((name, mean_curve, se_curve))

    # Print final gap table
    print("\nBranin BO assumption sweep (20 runs):")
    for name, mean_curve, se_curve in summary:
        print(f"  {name:<28} final gap @35: {mean_curve[-1]:.3f} ± {se_curve[-1]:.3f}")

    # Plot learning curves
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(1, 36)
    for name, mean_curve, se_curve in summary:
        ax.plot(x, mean_curve, linewidth=2, label=name)
        ax.fill_between(
            x,
            np.clip(mean_curve - se_curve, 0, 1),
            np.clip(mean_curve + se_curve, 0, 1),
            alpha=0.2,
        )

    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Gap (higher is better)")
    ax.set_title("Branin learning curves: baseline vs improved assumptions")
    ax.set_xlim(1, 35)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(save_dir, "branin_bo_learning_comparison.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

