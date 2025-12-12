"""
Explorations: visual comparison of baseline SE GP vs SE+Periodic(x1) GP on Branin.

Run: python branin_visual_comparison.py
Outputs:
  - baseline_vs_selected_branin_heatmaps_grid.png
  - baseline_vs_selected_branin_zscore_comparison.png
  - baseline_branin_se_heatmaps.png
  - selected_branin_se_plus_periodic_x1_heatmaps.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import ConstantKernel, RBF, ExpSineSquared

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

from model_fitting import (  # noqa: E402
    branin,
    generate_sobol_points,
    fit_gp,
    ActiveDimKernel,
)


def predict_on_grid(gp, bounds, n=100):
    x1 = np.linspace(bounds[0][0], bounds[0][1], n)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    mean, std = gp.predict(X_grid, return_std=True)
    return X1, X2, mean.reshape(X1.shape), std.reshape(X1.shape)


def zscores(gp, bounds, n=60):
    x1 = np.linspace(bounds[0][0], bounds[0][1], n)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])
    y_true = branin(X1, X2).ravel()
    mu, sigma = gp.predict(X_test, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    return (y_true - mu) / sigma


def plot_heatmaps(X_train, y_train, gp, bounds, title, save_path):
    X1, X2, mu, sigma = predict_on_grid(gp, bounds, n=120)
    y_true = branin(X1, X2)
    resid = mu - y_true

    vmin_mu, vmax_mu = y_true.min(), y_true.max()
    vmax_res = np.max(np.abs(resid))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    im0 = axes[0].imshow(mu, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
                         origin="lower", aspect="auto", cmap="viridis",
                         vmin=vmin_mu, vmax=vmax_mu)
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c="red", s=25, edgecolor="white")
    axes[0].set_title("Posterior mean")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(sigma, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
                         origin="lower", aspect="auto", cmap="plasma")
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c="white", s=25, edgecolor="black")
    axes[1].set_title("Posterior std")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(resid, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
                         origin="lower", aspect="auto", cmap="RdBu_r",
                         vmin=-vmax_res, vmax=vmax_res)
    axes[2].set_title("Residual μ(x) − f(x)")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    bounds = [(-5, 10), (0, 15)]

    X_train = generate_sobol_points(32, bounds, seed=42)
    y_train = branin(X_train[:, 0], X_train[:, 1])

    rbf = RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))

    baseline_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * rbf
    baseline_gp = fit_gp(X_train, y_train, baseline_kernel, noise_level=0.001, normalize_y=True, n_restarts=15)

    periodic_x1 = ActiveDimKernel(
        ExpSineSquared(length_scale=1.0, periodicity=2 * np.pi),
        active_dims=[0],
    )
    periodic_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * (rbf + periodic_x1)
    periodic_gp = fit_gp(X_train, y_train, periodic_kernel, noise_level=0.001, normalize_y=True, n_restarts=20)

    # Individual heatmaps
    plot_heatmaps(
        X_train, y_train, baseline_gp, bounds,
        "Baseline SE (ARD) GP",
        os.path.join(save_dir, "baseline_branin_se_heatmaps.png"),
    )
    plot_heatmaps(
        X_train, y_train, periodic_gp, bounds,
        "Improved SE + Periodic(x1) GP",
        os.path.join(save_dir, "selected_branin_se_plus_periodic_x1_heatmaps.png"),
    )

    # Combined 2x3 comparison
    X1, X2, mu_b, std_b = predict_on_grid(baseline_gp, bounds, n=120)
    _, _, mu_p, std_p = predict_on_grid(periodic_gp, bounds, n=120)
    y_true = branin(X1, X2)
    res_b = mu_b - y_true
    res_p = mu_p - y_true

    vmin_mu, vmax_mu = y_true.min(), y_true.max()
    vmax_std = max(std_b.max(), std_p.max())
    vmax_res = max(np.max(np.abs(res_b)), np.max(np.abs(res_p)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    rows = [
        ("Baseline SE", mu_b, std_b, res_b),
        ("SE + Periodic(x1)", mu_p, std_p, res_p),
    ]

    for r, (label, mu, sigma, resid) in enumerate(rows):
        im0 = axes[r, 0].imshow(mu, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
                                origin="lower", aspect="auto", cmap="viridis",
                                vmin=vmin_mu, vmax=vmax_mu)
        axes[r, 0].scatter(X_train[:, 0], X_train[:, 1], c="red", s=18, edgecolor="white")
        axes[r, 0].set_title(f"{label}: mean")

        im1 = axes[r, 1].imshow(sigma, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
                                origin="lower", aspect="auto", cmap="plasma",
                                vmin=0, vmax=vmax_std)
        axes[r, 1].scatter(X_train[:, 0], X_train[:, 1], c="white", s=18, edgecolor="black")
        axes[r, 1].set_title(f"{label}: std")

        im2 = axes[r, 2].imshow(resid, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
                                origin="lower", aspect="auto", cmap="RdBu_r",
                                vmin=-vmax_res, vmax=vmax_res)
        axes[r, 2].set_title(f"{label}: residual")

        for c in range(3):
            axes[r, c].set_xlabel("$x_1$")
            axes[r, c].set_ylabel("$x_2$")

    plt.colorbar(im0, ax=axes[:, 0], fraction=0.025, pad=0.02, label="μ(x)")
    plt.colorbar(im1, ax=axes[:, 1], fraction=0.025, pad=0.02, label="σ(x)")
    plt.colorbar(im2, ax=axes[:, 2], fraction=0.025, pad=0.02, label="μ(x) − f(x)")

    plt.suptitle("Baseline vs Improved GP on Branin (original scale)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "baseline_vs_selected_branin_heatmaps_grid.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # Z-score KDE comparison
    z_b = zscores(baseline_gp, bounds)
    z_p = zscores(periodic_gp, bounds)

    xs = np.linspace(-5, 5, 300)
    from scipy.stats import gaussian_kde, norm  # local import

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    kde_b = gaussian_kde(z_b[np.abs(z_b) < 10])
    kde_p = gaussian_kde(z_p[np.abs(z_p) < 10])
    ax.plot(xs, kde_b(xs), label="Baseline SE", color="tab:blue")
    ax.plot(xs, kde_p(xs), label="SE + Periodic(x1)", color="tab:green")
    ax.plot(xs, norm.pdf(xs), "k--", label="N(0,1)")
    ax.set_xlim(-5, 5)
    ax.set_xlabel("z-score")
    ax.set_ylabel("density")
    ax.set_title("Z-score calibration comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "baseline_vs_selected_branin_zscore_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close()

    print("Saved comparison plots to explorations/.")


if __name__ == "__main__":
    main()
