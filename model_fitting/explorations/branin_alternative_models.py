"""
Explorations: alternative GP models for Branin.

Goal: question baseline assumptions (noise scaling, kernel family) and
try richer kernels that may improve fit/calibration on the original scale.

Run: python branin_alternative_models.py
Outputs: printed table of RMSE/NLPD/z-score coverage for each model.
"""

import os
import sys
import numpy as np

from sklearn.gaussian_process.kernels import (
    Kernel, ConstantKernel, RBF, Matern, RationalQuadratic,
    ExpSineSquared, DotProduct
)

# Import shared helpers from parent script
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
from model_fitting import branin, generate_sobol_points, fit_gp  # noqa: E402


class ActiveDimKernel(Kernel):
    """Kernel wrapper that applies base_kernel to selected input dimensions."""

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


def make_test_grid(bounds, n=80):
    x1 = np.linspace(bounds[0][0], bounds[0][1], n)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])
    y_true = branin(X1, X2).ravel()
    return X_test, y_true


def score_model(gp, X_test, y_true):
    y_mean, y_std = gp.predict(X_test, return_std=True)
    y_std = np.maximum(y_std, 1e-9)

    rmse = float(np.sqrt(np.mean((y_mean - y_true) ** 2)))

    nlpd = 0.5 * np.log(2 * np.pi * y_std ** 2) + 0.5 * ((y_true - y_mean) ** 2) / (y_std ** 2)
    nlpd = float(np.mean(nlpd))

    z = (y_true - y_mean) / y_std
    z_mean = float(np.mean(z))
    z_std = float(np.std(z))
    cov1 = float(np.mean(np.abs(z) <= 1) * 100)
    cov2 = float(np.mean(np.abs(z) <= 2) * 100)

    return rmse, nlpd, z_mean, z_std, cov1, cov2


def main():
    np.random.seed(42)
    bounds = [(-5, 10), (0, 15)]

    X_train = generate_sobol_points(32, bounds, seed=42)
    y_train = branin(X_train[:, 0], X_train[:, 1])

    X_test, y_true = make_test_grid(bounds, n=100)

    # Baseline pieces
    rbf_ard = RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
    matern52_ard = Matern(length_scale=[1.0, 1.0], nu=2.5, length_scale_bounds=(1e-2, 1e2))
    rq_iso = RationalQuadratic(length_scale=1.0, alpha=1.0)
    linear = DotProduct(sigma_0=1.0)

    # Periodic in x1 only via ActiveDimKernel
    periodic_x1 = ActiveDimKernel(
        ExpSineSquared(length_scale=1.0, periodicity=2 * np.pi),
        active_dims=[0],
    )

    kernels = {
        "SE (ARD)": ConstantKernel(1.0, (1e-3, 1e3)) * rbf_ard,
        "Matern 5/2 (ARD)": ConstantKernel(1.0, (1e-3, 1e3)) * matern52_ard,
        "RationalQuadratic": ConstantKernel(1.0, (1e-3, 1e3)) * rq_iso,
        "SE + Linear": ConstantKernel(1.0, (1e-3, 1e3)) * (rbf_ard + linear),
        "SE + Periodic(x1)": ConstantKernel(1.0, (1e-3, 1e3)) * (rbf_ard + periodic_x1),
        "SE × Periodic(x1)": ConstantKernel(1.0, (1e-3, 1e3)) * (rbf_ard * periodic_x1),
    }

    results = []
    print("\nFitting alternative models on ORIGINAL Branin scale (noise σ=0.001)...\n")

    for name, kernel in kernels.items():
        gp = fit_gp(X_train, y_train, kernel, noise_level=0.001, normalize_y=True, n_restarts=15)
        rmse, nlpd, z_mean, z_std, cov1, cov2 = score_model(gp, X_test, y_true)
        results.append((name, rmse, nlpd, z_mean, z_std, cov1, cov2, str(gp.kernel_)))

    results.sort(key=lambda r: r[1])

    header = f"{'Model':<20} {'RMSE':>8} {'NLPD':>8} {'z-mean':>8} {'z-std':>8} {'cov1%':>7} {'cov2%':>7}"
    print(header)
    print("-" * len(header))
    for name, rmse, nlpd, z_mean, z_std, cov1, cov2, fitted in results:
        print(f"{name:<20} {rmse:8.2f} {nlpd:8.2f} {z_mean:8.3f} {z_std:8.3f} {cov1:7.1f} {cov2:7.1f}")
        print(f"  fitted kernel: {fitted}")


if __name__ == "__main__":
    main()

