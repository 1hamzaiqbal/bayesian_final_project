"""
Explorations: sweep modeling assumptions for acquisition-function comparisons.

We revisit the bonus acquisition study with alternative GP modeling choices:
  - log(y+1) vs original scale (Branin)
  - RBF vs Matern 3/2 (LDA/SVM)
  - SE + Periodic(x1) for Branin (structure-matched)
  - corrected noise scaling under normalize_y=True

Outputs (saved in this folder):
  - modeling_sweep_final_gap.png
  - modeling_sweep_auc.png

Run (recommended first): python acquisition_modeling_sweep.py --n-runs 10
Run (final):             python acquisition_modeling_sweep.py --n-runs 20
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel,
    ConstantKernel,
    ExpSineSquared,
    Matern,
    RBF,
)

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
import bonus_acquisition_comparison as base  # noqa: E402


class ActiveDimKernel(Kernel):
    """Apply a base kernel to selected input dimensions."""

    def __init__(self, base_kernel: Kernel, active_dims: List[int]):
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
        return ActiveDimKernel(self.base_kernel.clone_with_theta(theta), list(self.active_dims))


@dataclass(frozen=True)
class ModelingConfig:
    name: str
    kernel_fn: Callable[[str], Kernel]
    use_log_transform: Callable[[str], bool]
    corrected_noise: bool
    n_restarts: int


def fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Kernel,
    noise_level: float,
    corrected_noise: bool,
    n_restarts: int,
) -> GaussianProcessRegressor:
    alpha = noise_level**2
    if corrected_noise:
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


def run_bo_with_acq(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    acq_name: str,
    acq_params: Dict[str, float],
    kernel: Kernel,
    use_log_transform: bool,
    corrected_noise: bool,
    n_restarts: int,
    init_indices: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n_pool = len(X_pool)
    available = np.ones(n_pool, dtype=bool)

    selected_indices: List[int] = []
    y_values: List[float] = []
    best_so_far: List[float] = []

    for idx in init_indices:
        idx_int = int(idx)
        selected_indices.append(idx_int)
        y_values.append(float(y_pool[idx_int]))
        available[idx_int] = False
        best_so_far.append(float(np.min(y_values)))

    for _ in range(30):
        X_train = X_pool[selected_indices]
        y_train = np.array(y_values)

        if use_log_transform:
            y_fit = np.log(y_train + 1)
        else:
            y_fit = y_train

        gp = fit_gp(
            X_train,
            y_fit,
            kernel=kernel,
            noise_level=0.001,
            corrected_noise=corrected_noise,
            n_restarts=n_restarts,
        )

        avail_idx = np.where(available)[0]
        mu, sigma = gp.predict(X_pool[avail_idx], return_std=True)

        if use_log_transform:
            f_best = float(np.log(np.min(y_values) + 1))
        else:
            f_best = float(np.min(y_values))

        if acq_name == "Random":
            next_global = int(rng.choice(avail_idx))
        elif acq_name == "EI":
            acq = base.expected_improvement(mu, sigma, f_best, xi=acq_params.get("xi", 0.01))
            next_global = int(avail_idx[int(np.argmax(acq))])
        elif acq_name == "PI":
            acq = base.probability_of_improvement(mu, sigma, f_best, xi=acq_params.get("xi", 0.01))
            next_global = int(avail_idx[int(np.argmax(acq))])
        elif acq_name == "LCB":
            acq = base.lower_confidence_bound(mu, sigma, kappa=acq_params.get("kappa", 2.0))
            next_global = int(avail_idx[int(np.argmax(acq))])
        else:
            raise ValueError(f"Unknown acquisition: {acq_name}")

        selected_indices.append(next_global)
        y_values.append(float(y_pool[next_global]))
        available[next_global] = False
        best_so_far.append(float(np.min(y_values)))

    return {
        "best_so_far": np.array(best_so_far),
        "init_indices": init_indices.copy(),
    }


def gap_curve(best_so_far: np.ndarray, f_opt: float) -> np.ndarray:
    f_best_initial = float(best_so_far[4])
    gaps = []
    for best in best_so_far:
        gaps.append(base.compute_gap(float(best), f_best_initial, f_opt))
    return np.array(gaps)


def auc_from_gap(gaps: np.ndarray) -> float:
    # Average gap from obs 5..35 (inclusive)
    return float(np.mean(gaps[4:]))


def load_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
    project_root = os.path.dirname(PARENT_DIR)
    lda_data = np.loadtxt(os.path.join(project_root, "lda.csv"), delimiter=",")
    svm_data = np.loadtxt(os.path.join(project_root, "svm.csv"), delimiter=",")

    sampler = qmc.Sobol(d=2, scramble=True, seed=123)
    X_branin = sampler.random(1000)
    X_branin = qmc.scale(X_branin, [-5, 0], [10, 15])
    y_branin = base.branin(X_branin[:, 0], X_branin[:, 1])

    return {
        "Branin": (X_branin, y_branin, 0.397887),
        "LDA": (lda_data[:, :3], lda_data[:, 3], float(lda_data[:, 3].min())),
        "SVM": (svm_data[:, :3], svm_data[:, 3], float(svm_data[:, 3].min())),
    }


def make_modeling_configs() -> List[ModelingConfig]:
    def kernel_baseline(dataset: str) -> Kernel:
        if dataset == "Branin":
            return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2)
            )
        return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2)
        )

    def kernel_improved(dataset: str) -> Kernel:
        if dataset == "Branin":
            rbf = RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
            periodic_x1 = ActiveDimKernel(
                ExpSineSquared(length_scale=1.0, periodicity=2 * np.pi),
                active_dims=[0],
            )
            return ConstantKernel(1.0, (1e-3, 1e3)) * (rbf + periodic_x1)
        return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=1.5
        )

    def use_log_baseline(dataset: str) -> bool:
        # Matches the existing bonus writeup: log(y+1) everywhere
        return True

    def use_log_improved(dataset: str) -> bool:
        # Branin: stay on original scale (better calibrated in model fitting)
        # LDA/SVM: keep log(y+1) to stabilize scale
        return dataset != "Branin"

    return [
        ModelingConfig(
            name="Baseline (log + RBF)",
            kernel_fn=kernel_baseline,
            use_log_transform=use_log_baseline,
            corrected_noise=False,
            n_restarts=5,
        ),
        ModelingConfig(
            name="Baseline + corrected noise",
            kernel_fn=kernel_baseline,
            use_log_transform=use_log_baseline,
            corrected_noise=True,
            n_restarts=5,
        ),
        ModelingConfig(
            name="Improved (Branin periodic; LDA/SVM Matern)",
            kernel_fn=kernel_improved,
            use_log_transform=use_log_improved,
            corrected_noise=True,
            n_restarts=8,
        ),
    ]


def plot_bars(
    metrics: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    metric_key: str,
    save_path: str,
):
    # metrics[dataset][config][method] = per-run metric values
    datasets = list(metrics.keys())
    configs = list(next(iter(metrics.values())).keys())
    methods = list(next(iter(next(iter(metrics.values())).values())).keys())

    fig, axes = plt.subplots(1, len(datasets), figsize=(17, 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        x = np.arange(len(methods))
        width = 0.25

        for i, config in enumerate(configs):
            vals = np.array([metrics[dataset][config][m] for m in methods])
            means = vals.mean(axis=1)
            ses = vals.std(axis=1) / np.sqrt(vals.shape[1])
            ax.bar(
                x + (i - (len(configs) - 1) / 2) * width,
                means,
                width,
                yerr=ses,
                capsize=4,
                label=config,
                alpha=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_title(dataset)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel(metric_key)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.05), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    save_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = load_datasets()

    # Keep the same set of acquisitions as the main bonus section
    methods = {
        "Random": ("Random", {}),
        "EI": ("EI", {"xi": 0.01}),
        "PI": ("PI", {"xi": 0.01}),
        "LCB κ=1": ("LCB", {"kappa": 1.0}),
        "LCB κ=2": ("LCB", {"kappa": 2.0}),
    }

    configs = make_modeling_configs()

    final_gap: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    auc: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for dataset_name, (X_pool, y_pool, f_opt) in datasets.items():
        n_pool = len(X_pool)
        all_init_indices = []
        for run in range(args.n_runs):
            run_rng = np.random.default_rng(seed=run * 42 + 1)
            all_init_indices.append(run_rng.choice(n_pool, size=5, replace=False))

        final_gap[dataset_name] = {}
        auc[dataset_name] = {}

        for cfg in configs:
            final_gap[dataset_name][cfg.name] = {}
            auc[dataset_name][cfg.name] = {}

            kernel = cfg.kernel_fn(dataset_name)
            use_log = cfg.use_log_transform(dataset_name)

            for method_label, (acq_name, acq_params) in methods.items():
                gaps = []
                aucs = []
                for run in range(args.n_runs):
                    init_indices = all_init_indices[run].copy()
                    rng = np.random.default_rng(seed=run * 999 + 3)

                    hist = run_bo_with_acq(
                        X_pool,
                        y_pool,
                        acq_name=acq_name,
                        acq_params=acq_params,
                        kernel=kernel,
                        use_log_transform=use_log,
                        corrected_noise=cfg.corrected_noise,
                        n_restarts=cfg.n_restarts,
                        init_indices=init_indices,
                        rng=rng,
                    )

                    assert np.array_equal(hist["init_indices"], init_indices)
                    gaps_curve = gap_curve(hist["best_so_far"], f_opt=f_opt)
                    gaps.append(gaps_curve[-1])
                    aucs.append(auc_from_gap(gaps_curve))

                final_gap[dataset_name][cfg.name][method_label] = np.array(gaps)
                auc[dataset_name][cfg.name][method_label] = np.array(aucs)

            # Print quick summary per dataset/config
            print(f"\n{dataset_name} — {cfg.name}")
            for method_label in methods:
                vals = final_gap[dataset_name][cfg.name][method_label]
                print(f"  {method_label:<8} final gap: {vals.mean():.3f} ± {vals.std()/np.sqrt(len(vals)):.3f}")

    plot_bars(
        final_gap,
        metric_key="Final gap (@35)",
        save_path=os.path.join(save_dir, "modeling_sweep_final_gap.png"),
    )
    plot_bars(
        auc,
        metric_key="AUC(mean gap, obs 5–35)",
        save_path=os.path.join(save_dir, "modeling_sweep_auc.png"),
    )

    print("\nSaved plots to bonus/explorations/.")


if __name__ == "__main__":
    main()
