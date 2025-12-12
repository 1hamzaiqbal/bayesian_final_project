"""
Model Fitting for Bayesian Optimization Project

This script addresses all bullet points from the model fitting section:
1. Generate 32 Sobol training points for Branin
2. Fit GP with constant mean + SE kernel
3. Report hyperparameters
4. Heatmaps of posterior mean and std
5. Z-score calibration analysis
6. Log transformation comparison
7. BIC computation
8. Model search with different kernels
9. Apply to SVM and LDA datasets

Author: Generated for Bayesian Optimization final project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import qmc
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel, RBF, ConstantKernel, Matern, RationalQuadratic,
    WhiteKernel, ExpSineSquared, DotProduct
)

# Add parent directory to path to import branin function
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def branin(x1, x2, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    """Branin function."""
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    return term1 + term2 + s


def generate_sobol_points(n_points, bounds, seed=42):
    """
    Generate Sobol sequence points within given bounds.
    
    Parameters:
    -----------
    n_points : int
        Number of points to generate
    bounds : list of tuples
        [(min1, max1), (min2, max2), ...]
    seed : int
        Random seed for reproducibility
    """
    dim = len(bounds)
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    
    # Generate points in [0, 1]^d
    points = sampler.random(n_points)
    
    # Scale to bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    scaled_points = qmc.scale(points, lower, upper)
    
    return scaled_points


def fit_gp(X, y, kernel, noise_level=0.001, n_restarts=10, normalize_y=True):
    """
    Fit a Gaussian process with given kernel.
    
    Parameters:
    -----------
    X : array, shape (n, d)
        Training inputs
    y : array, shape (n,)
        Training outputs
    kernel : sklearn kernel
        Covariance function
    noise_level : float
        Fixed noise level (alpha parameter)
    n_restarts : int
        Number of optimizer restarts
        
    Returns:
    --------
    gp : fitted GaussianProcessRegressor
    """
    # sklearn rescales y internally when normalize_y=True. To keep the noise
    # level interpretable in original output units, scale alpha accordingly.
    alpha = noise_level**2  # alpha is variance, not std
    if normalize_y:
        y_std = np.std(y)
        if y_std > 0:
            alpha = (noise_level / y_std) ** 2

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts,
        normalize_y=normalize_y,
        random_state=42
    )
    gp.fit(X, y)
    return gp


def compute_bic(gp, X, y, include_mean=True):
    """
    Compute Bayesian Information Criterion.
    
    BIC = k * log(n) - 2 * log_likelihood
    
    where k = number of hyperparameters, n = number of observations
    """
    log_likelihood = gp.log_marginal_likelihood_value_
    n = len(y)
    
    # Count hyperparameters from kernel (+ mean if learned)
    k = gp.kernel_.n_dims + (1 if include_mean else 0)
    
    bic = k * np.log(n) - 2 * log_likelihood
    return bic, log_likelihood, k


def create_posterior_heatmaps(gp, X_train, y_train, bounds, title_prefix="", save_dir=None, save_filename=None):
    """Create heatmaps of GP posterior mean and std deviation."""
    # Create prediction grid
    x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Get predictions
    y_mean, y_std = gp.predict(X_grid, return_std=True)
    y_mean = y_mean.reshape(X1.shape)
    y_std = y_std.reshape(X1.shape)
    
    # True function values
    y_true = branin(X1, X2)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # True function
    im0 = axes[0].imshow(y_true, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], 
                         origin='lower', aspect='auto', cmap='viridis')
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c='red', s=30, edgecolor='white', 
                    label='Training points', zorder=5)
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].set_title('True Branin Function')
    plt.colorbar(im0, ax=axes[0], label='$f(x)$')
    axes[0].legend(loc='upper right')
    
    # Posterior mean
    im1 = axes[1].imshow(y_mean, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], 
                         origin='lower', aspect='auto', cmap='viridis')
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c='red', s=30, edgecolor='white', zorder=5)
    axes[1].set_xlabel('$x_1$')
    axes[1].set_ylabel('$x_2$')
    axes[1].set_title('GP Posterior Mean')
    plt.colorbar(im1, ax=axes[1], label='$\\mu(x)$')
    
    # Posterior std
    im2 = axes[2].imshow(y_std, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], 
                         origin='lower', aspect='auto', cmap='plasma')
    axes[2].scatter(X_train[:, 0], X_train[:, 1], c='white', s=30, edgecolor='black', zorder=5)
    axes[2].set_xlabel('$x_1$')
    axes[2].set_ylabel('$x_2$')
    axes[2].set_title('GP Posterior Std Dev')
    plt.colorbar(im2, ax=axes[2], label='$\\sigma(x)$')
    
    plt.suptitle(f'{title_prefix}GP Posterior on Branin Function (32 Sobol training points)', fontsize=12)
    plt.tight_layout()
    
    if save_dir:
        filename = save_filename or f"{title_prefix.lower().replace(' ', '_')}posterior_heatmaps.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.close()
    
    return y_mean, y_std, y_true


def create_residual_heatmap(gp, bounds, title_prefix="", save_dir=None, use_log_transform=False, save_filename=None):
    """Create a heatmap of residuals μ(x) - f(x) to show systematic errors."""
    # Create prediction grid
    x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Get predictions
    y_mean = gp.predict(X_grid, return_std=False)
    y_mean = y_mean.reshape(X1.shape)
    
    # True function values
    y_true = branin(X1, X2)
    if use_log_transform:
        y_true = np.log(y_true + 1)
    
    # Residuals
    residuals = y_mean - y_true
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use diverging colormap centered at 0
    vmax = max(abs(residuals.min()), abs(residuals.max()))
    im = ax.imshow(residuals, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], 
                   origin='lower', aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    
    ax.set_xlabel('$x_1$', fontsize=11)
    ax.set_ylabel('$x_2$', fontsize=11)
    ax.set_title(f'{title_prefix}Residuals: $\\mu(x) - f(x)$\n(Red = overprediction, Blue = underprediction)', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Residual', fontsize=10)
    
    # Add statistics
    stats_text = f'RMSE: {np.sqrt(np.mean(residuals**2)):.2f}\n'
    stats_text += f'Max error: {np.max(np.abs(residuals)):.2f}\n'
    stats_text += f'Mean error: {np.mean(residuals):.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            ha='left', va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        filename = save_filename or f"{title_prefix.lower().replace(' ', '_')}residual_heatmap.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.close()
    
    return residuals


def compute_zscores(gp, X_train, use_log_transform=False):
    """
    Compute z-scores for calibration check.
    
    For a well-calibrated GP, z-scores should be approximately standard normal.
    """
    # Create dense test grid
    x1 = np.linspace(-5, 10, 50)
    x2 = np.linspace(0, 15, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Get predictions
    y_mean, y_std = gp.predict(X_test, return_std=True)
    
    # Get true values
    y_true = branin(X1, X2).ravel()
    
    if use_log_transform:
        y_true = np.log(y_true + 1)
    
    # Z-scores: (y_true - y_pred) / sigma
    # Avoid division by near-zero
    y_std = np.maximum(y_std, 1e-6)
    z_scores = (y_true - y_mean) / y_std
    
    return z_scores


def plot_zscore_kde(z_scores, title, save_path=None):
    """Plot KDE of z-scores compared to standard normal, with coverage metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out extreme values for better visualization
    z_filtered = z_scores[np.abs(z_scores) < 10]
    
    # KDE of z-scores
    x = np.linspace(-5, 5, 200)
    if len(z_filtered) > 1:
        kde = stats.gaussian_kde(z_filtered)
        ax.fill_between(x, kde(x), alpha=0.5, color='steelblue', label='Empirical z-scores')
        ax.plot(x, kde(x), color='steelblue', linewidth=2)
    
    # Standard normal for reference
    normal_pdf = stats.norm.pdf(x)
    ax.plot(x, normal_pdf, 'r--', linewidth=2, label='Standard Normal N(0,1)')
    
    ax.set_xlabel('Z-score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.set_xlim(-5, 5)
    
    # Compute coverage metrics
    n_total = len(z_scores)
    cov_1 = np.mean(np.abs(z_scores) <= 1) * 100  # Should be ~68.3%
    cov_2 = np.mean(np.abs(z_scores) <= 2) * 100  # Should be ~95.4%
    cov_3 = np.mean(np.abs(z_scores) <= 3) * 100  # Should be ~99.7%
    
    # Add statistics and coverage
    mean_z = np.mean(z_filtered)
    std_z = np.std(z_filtered)
    stats_text = f'Mean: {mean_z:.3f}\nStd: {std_z:.3f}\n\n'
    stats_text += f'Coverage:\n'
    stats_text += f'  |z|≤1: {cov_1:.1f}% (target: 68.3%)\n'
    stats_text += f'  |z|≤2: {cov_2:.1f}% (target: 95.4%)\n'
    stats_text += f'  |z|≤3: {cov_3:.1f}% (target: 99.7%)'
    
    ax.text(0.98, 0.98, stats_text, 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.basename(save_path)}")
    
    plt.close()
    
    return mean_z, std_z, (cov_1, cov_2, cov_3)


def search_kernels(X, y, noise_level=0.001, verbose=True):
    """
    Search over different kernels to find the best BIC score.
    
    Returns list of (kernel_name, bic, log_likelihood, n_params, fitted_gp)
    """
    n_dim = X.shape[1]
    
    # Define kernels to try - use appropriate dimensions
    kernels = {
        'SE (RBF)': ConstantKernel(1.0) * RBF(length_scale=[1.0]*n_dim),
        'Matern 3/2': ConstantKernel(1.0) * Matern(length_scale=[1.0]*n_dim, nu=1.5),
        'Matern 5/2': ConstantKernel(1.0) * Matern(length_scale=[1.0]*n_dim, nu=2.5),
        'RationalQuadratic': ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0),
        'SE + Matern 5/2': ConstantKernel(1.0) * RBF(length_scale=1.0) + ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
        'SE (isotropic)': ConstantKernel(1.0) * RBF(length_scale=1.0),
    }

    # Branin has a clear periodic component in x1 (cos term) and a global trend;
    # try richer compositional kernels in 2D only.
    if n_dim == 2:
        periodic_x1 = ActiveDimKernel(
            ExpSineSquared(length_scale=1.0, periodicity=2 * np.pi),
            active_dims=[0],
        )
        kernels.update({
            'Periodic (isotropic)': ConstantKernel(1.0) * ExpSineSquared(length_scale=1.0, periodicity=2*np.pi),
            'SE + Periodic': ConstantKernel(1.0) * (RBF(length_scale=[1.0]*n_dim) + ExpSineSquared(length_scale=1.0, periodicity=2*np.pi)),
            'SE × Periodic': ConstantKernel(1.0) * (RBF(length_scale=[1.0]*n_dim) * ExpSineSquared(length_scale=1.0, periodicity=2*np.pi)),
            'SE + Periodic(x1)': ConstantKernel(1.0) * (RBF(length_scale=[1.0]*n_dim) + periodic_x1),
            'SE × Periodic(x1)': ConstantKernel(1.0) * (RBF(length_scale=[1.0]*n_dim) * periodic_x1),
            'SE + Linear': ConstantKernel(1.0) * (RBF(length_scale=[1.0]*n_dim) + DotProduct(sigma_0=1.0)),
        })
    
    results = []
    
    for name, kernel in kernels.items():
        try:
            gp = fit_gp(X, y, kernel, noise_level=noise_level, normalize_y=True)
            bic, ll, k = compute_bic(gp, X, y)
            results.append({
                'name': name,
                'bic': bic,
                'log_likelihood': ll,
                'n_params': k,
                'gp': gp,
                'kernel': str(gp.kernel_)
            })
            if verbose:
                print(f"  {name}: BIC = {bic:.2f}, LL = {ll:.2f}, k = {k}")
        except Exception as e:
            if verbose:
                print(f"  {name}: FAILED - {e}")
    
    # Sort by BIC (lower is better)
    results.sort(key=lambda x: x['bic'])
    
    return results


def analyze_branin_original(save_dir):
    """
    Part 1: Branin function with original (untransformed) data.
    """
    print("=" * 70)
    print("PART 1: BRANIN FUNCTION - ORIGINAL DATA")
    print("=" * 70)
    
    # Generate 32 Sobol points
    bounds = [(-5, 10), (0, 15)]
    X_train = generate_sobol_points(32, bounds, seed=42)
    y_train = branin(X_train[:, 0], X_train[:, 1])
    
    print(f"\nGenerated {len(X_train)} Sobol training points")
    print(f"Domain: x1 ∈ [{bounds[0][0]}, {bounds[0][1]}], x2 ∈ [{bounds[1][0]}, {bounds[1][1]}]")
    print(f"y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"y mean: {y_train.mean():.2f}, y std: {y_train.std():.2f}")
    
    # Fit GP with SE kernel
    print("\n--- Fitting GP with Constant Mean + ARD SE Kernel ---")
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
    gp = fit_gp(X_train, y_train, kernel, noise_level=0.001, normalize_y=True)
    
    # Report hyperparameters including mean
    print("\nLearned Hyperparameters:")
    print(f"  Kernel: {gp.kernel_}")
    print(f"  Log-marginal-likelihood: {gp.log_marginal_likelihood_value_:.4f}")
    
    # Report constant mean (sklearn uses normalize_y=True, so mean is stored internally)
    learned_mean = gp._y_train_mean
    print(f"  Constant mean (from normalize_y): {learned_mean:.2f}")
    print(f"  → This should be near the empirical mean of y ({y_train.mean():.2f}) ✓")
    
    # Extract length scales
    kernel_params = gp.kernel_.get_params()
    print(f"\n  Interpretation:")
    print(f"    - The output scale (constant kernel) controls the overall variance")
    print(f"    - The length scales control how quickly correlation decays with distance")
    print(f"    - Given domain: x1 ∈ [-5, 10] (width 15), x2 ∈ [0, 15] (width 15)")
    
    # Acknowledge large ℓ₂ issue if present
    try:
        length_scales = np.atleast_1d(gp.kernel_.k2.length_scale)
        if len(length_scales) >= 2:
            l2 = float(length_scales[1])
            x2_width = bounds[1][1] - bounds[1][0]
            if l2 > 2 * x2_width:
                print(f"\n  ⚠ NOTE: ℓ₂ ≈ {l2:.2f} is > 2× the domain width [{bounds[1][0]}, {bounds[1][1]}].")
                print(f"    This suggests the model thinks f varies slowly in x2,")
                print(f"    possibly indicating local optimum in MLL optimization or misspecification.")
                print(f"    The Branin function has a quadratic dependence on x2, so this may oversmooth.")
    except Exception:
        pass
    
    # Create heatmaps
    print("\n--- Creating Posterior Heatmaps ---")
    y_mean, y_std, y_true = create_posterior_heatmaps(
        gp,
        X_train,
        y_train,
        bounds,
        title_prefix="Baseline SE (original scale) ",
        save_dir=save_dir,
        save_filename="baseline_branin_se_posterior_heatmaps.png",
    )
    
    # Create residual heatmap
    print("\n--- Creating Residual Heatmap ---")
    create_residual_heatmap(
        gp,
        bounds,
        title_prefix="Baseline SE (original scale) ",
        save_dir=save_dir,
        save_filename="baseline_branin_se_residual_heatmap.png",
    )
    
    # Check posterior std at training points
    # NOTE: sklearn's GP with normalize_y=True rescales σ by data std
    y_pred_train, y_std_train = gp.predict(X_train, return_std=True)
    print("\nPosterior std at training points:")
    print(f"  Min: {y_std_train.min():.6f}")
    print(f"  Max: {y_std_train.max():.6f}")
    print(f"  Mean: {y_std_train.mean():.6f}")
    print(f"  NOTE: σ(x) is PREDICTIVE std (includes noise variance).")
    print(f"  We scale alpha by y-std so the effective noise std stays ≈ 0.001 in original units.")
    
    # Z-score calibration
    print("\n--- Z-Score Calibration Analysis ---")
    z_scores = compute_zscores(gp, X_train, use_log_transform=False)
    mean_z, std_z, coverage = plot_zscore_kde(
        z_scores, 
        "Z-Scores: Baseline SE on Branin (original scale)",
        save_path=os.path.join(save_dir, "baseline_branin_se_zscore.png")
    )
    print(f"  Z-score mean: {mean_z:.3f} (should be ≈ 0)")
    print(f"  Z-score std: {std_z:.3f} (should be ≈ 1)")
    print(f"  Coverage |z|≤1: {coverage[0]:.1f}% (target: 68.3%)")
    print(f"  Coverage |z|≤2: {coverage[1]:.1f}% (target: 95.4%)")
    
    # Interpretation: check if std indicates underconfidence/overconfidence
    if std_z > 1.5:
        print(f"  → Model is OVERCONFIDENT (std > 1.5 means GP uncertainty is too narrow)")
    elif std_z < 0.7:
        print(f"  → Model is UNDERCONFIDENT (std < 0.7 means GP uncertainty is too wide)")
    elif abs(mean_z) > 0.5:
        print(f"  → Model has SYSTEMATIC BIAS (mean deviates from 0)")
    else:
        print(f"  → Model appears reasonably calibrated")
    
    return X_train, y_train, gp


def analyze_high_noise(X_train, y_train, save_dir):
    """
    Additional analysis: GP behavior at high noise level (noise=10).
    This demonstrates how the GP posterior degrades when noise is high.
    """
    print("\n" + "=" * 70)
    print("HIGH NOISE ANALYSIS (noise=10)")
    print("=" * 70)
    
    bounds = [(-5, 10), (0, 15)]
    
    # Fit GP with high noise
    print("\n--- Fitting GP with High Noise Level (σ=10) ---")
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
    gp_high_noise = fit_gp(X_train, y_train, kernel, noise_level=10.0, normalize_y=True)
    
    print(f"\nLearned Hyperparameters (high noise):")
    print(f"  Kernel: {gp_high_noise.kernel_}")
    print(f"  Log-marginal-likelihood: {gp_high_noise.log_marginal_likelihood_value_:.4f}")
    
    # Create prediction grid
    x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Get predictions
    y_mean, y_std = gp_high_noise.predict(X_grid, return_std=True)
    y_mean = y_mean.reshape(X1.shape)
    y_std = y_std.reshape(X1.shape)
    
    # True function values
    y_true = branin(X1, X2)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # True function
    im0 = axes[0].imshow(y_true, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], 
                         origin='lower', aspect='auto', cmap='viridis')
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c='red', s=30, edgecolor='white', 
                    label='Training points', zorder=5)
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].set_title('True Branin Function')
    plt.colorbar(im0, ax=axes[0], label='$f(x)$')
    axes[0].legend(loc='upper right')
    
    # Posterior mean
    im1 = axes[1].imshow(y_mean, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], 
                         origin='lower', aspect='auto', cmap='viridis')
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c='red', s=30, edgecolor='white', zorder=5)
    axes[1].set_xlabel('$x_1$')
    axes[1].set_ylabel('$x_2$')
    axes[1].set_title('GP Posterior Mean (High Noise)')
    plt.colorbar(im1, ax=axes[1], label='$\\mu(x)$')
    
    # Posterior std
    im2 = axes[2].imshow(y_std, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], 
                         origin='lower', aspect='auto', cmap='plasma')
    axes[2].scatter(X_train[:, 0], X_train[:, 1], c='white', s=30, edgecolor='black', zorder=5)
    axes[2].set_xlabel('$x_1$')
    axes[2].set_ylabel('$x_2$')
    axes[2].set_title('GP Posterior Std Dev (High Noise)')
    plt.colorbar(im2, ax=axes[2], label='$\\sigma(x)$')
    
    plt.suptitle('GP Posterior with High Noise Level (σ=10)', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.join(save_dir, "explorations"), exist_ok=True)
    plt.savefig(os.path.join(save_dir, "explorations", "exp_high_noise_posterior_heatmaps.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: explorations/exp_high_noise_posterior_heatmaps.png")
    
    # Check posterior std at training points
    y_pred_train, y_std_train = gp_high_noise.predict(X_train, return_std=True)
    print(f"\nPosterior std at training points (high noise):")
    print(f"  Min: {y_std_train.min():.4f}")
    print(f"  Max: {y_std_train.max():.4f}")
    print(f"  Mean: {y_std_train.mean():.4f}")
    print(f"  → With high noise, std does NOT drop to near zero at training points")
    
    return gp_high_noise


def analyze_branin_log_transformed(X_train, save_dir):
    """
    Part 2: Branin function with log transformation.
    """
    print("\n" + "=" * 70)
    print("PART 2: BRANIN FUNCTION - LOG TRANSFORMED")
    print("=" * 70)
    
    bounds = [(-5, 10), (0, 15)]
    
    # Get original y values and transform
    y_original = branin(X_train[:, 0], X_train[:, 1])
    y_log = np.log(y_original + 1)
    
    print(f"\nTransformation: y' = log(y + 1)")
    print(f"Original y range: [{y_original.min():.2f}, {y_original.max():.2f}]")
    print(f"Log y range: [{y_log.min():.2f}, {y_log.max():.2f}]")
    
    # Fit GP
    print("\n--- Fitting GP with Log-Transformed Data ---")
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
    gp_log = fit_gp(X_train, y_log, kernel, noise_level=0.001, normalize_y=True)
    
    print(f"\nLearned Hyperparameters:")
    print(f"  Kernel: {gp_log.kernel_}")
    print(f"  Log-marginal-likelihood: {gp_log.log_marginal_likelihood_value_:.4f}")
    
    # Create modified heatmap for log-transformed
    x1 = np.linspace(-5, 10, 100)
    x2 = np.linspace(0, 15, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    y_mean, y_std = gp_log.predict(X_grid, return_std=True)
    y_mean = y_mean.reshape(X1.shape)
    y_std = y_std.reshape(X1.shape)
    
    # True log-transformed values
    y_true_log = np.log(branin(X1, X2) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    im0 = axes[0].imshow(y_true_log, extent=[-5, 10, 0, 15], origin='lower', aspect='auto', cmap='viridis')
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c='red', s=30, edgecolor='white', zorder=5)
    axes[0].set_xlabel('$x_1$'); axes[0].set_ylabel('$x_2$')
    axes[0].set_title('True: $\\log(f + 1)$')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(y_mean, extent=[-5, 10, 0, 15], origin='lower', aspect='auto', cmap='viridis')
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c='red', s=30, edgecolor='white', zorder=5)
    axes[1].set_xlabel('$x_1$'); axes[1].set_ylabel('$x_2$')
    axes[1].set_title('GP Posterior Mean')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(y_std, extent=[-5, 10, 0, 15], origin='lower', aspect='auto', cmap='plasma')
    axes[2].scatter(X_train[:, 0], X_train[:, 1], c='white', s=30, edgecolor='black', zorder=5)
    axes[2].set_xlabel('$x_1$'); axes[2].set_ylabel('$x_2$')
    axes[2].set_title('GP Posterior Std')
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle('Log-Transformed Branin: GP Posterior', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "baseline_branin_log_se_posterior_heatmaps.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: baseline_branin_log_se_posterior_heatmaps.png")
    
    # Z-score calibration
    print("\n--- Z-Score Calibration (Log-Transformed) ---")
    z_scores = compute_zscores(gp_log, X_train, use_log_transform=True)
    mean_z, std_z, coverage = plot_zscore_kde(
        z_scores,
        "Z-Scores: Log-Transformed Branin Data",
        save_path=os.path.join(save_dir, "baseline_branin_log_se_zscore.png")
    )
    print(f"  Z-score mean: {mean_z:.3f}")
    print(f"  Z-score std: {std_z:.3f}")
    print(f"  Coverage |z|≤1: {coverage[0]:.1f}% (target: 68.3%)")
    print(f"  Coverage |z|≤2: {coverage[1]:.1f}% (target: 95.4%)")
    
    # Interpretation
    if std_z > 1.5:
        print(f"  → Log-transformed model is OVERCONFIDENT (uncertainty too narrow)")
    else:
        print(f"  → Log-transformed model calibration is acceptable")
    
    return gp_log, y_log


def compute_bic_analysis(gp_log, y_log, X_train, save_dir):
    """
    Part 3: BIC computation and model search.
    """
    print("\n" + "=" * 70)
    print("PART 3: BIC COMPUTATION AND MODEL SEARCH")
    print("=" * 70)
    
    # BIC for the log-transformed SE model
    bic, ll, k = compute_bic(gp_log, X_train, y_log)
    print(f"\nBIC for Log-Transformed SE Model:")
    print(f"  Log-likelihood: {ll:.4f}")
    print(f"  Number of hyperparameters: {k}")
    print(f"  BIC = {k} × log(32) - 2 × {ll:.4f} = {bic:.4f}")
    
    # Model search
    print("\n--- Searching Over Different Kernels ---")
    print("(Using log-transformed Branin data)")
    results = search_kernels(X_train, y_log, noise_level=0.001)
    
    print("\n--- Model Rankings (by BIC, lower is better) ---")
    print(f"{'Rank':<5} {'Kernel':<25} {'BIC':<12} {'Log-Lik':<12} {'Params':<8}")
    print("-" * 65)
    for i, r in enumerate(results):
        print(f"{i+1:<5} {r['name']:<25} {r['bic']:<12.2f} {r['log_likelihood']:<12.2f} {r['n_params']:<8}")
    
    best = results[0]
    print(f"\n*** Best Model: {best['name']} ***")
    print(f"    BIC: {best['bic']:.4f}")
    print(f"    Fitted kernel: {best['kernel']}")
    
    return results


def analyze_real_benchmarks(save_dir):
    """
    Part 4: Model search for SVM and LDA benchmarks.
    """
    print("\n" + "=" * 70)
    print("PART 4: REAL BENCHMARKS (SVM AND LDA)")
    print("=" * 70)
    
    # Load data
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    lda_data = np.loadtxt(os.path.join(parent_dir, 'lda.csv'), delimiter=',')
    svm_data = np.loadtxt(os.path.join(parent_dir, 'svm.csv'), delimiter=',')
    
    np.random.seed(42)
    
    results_all = {}
    
    for name, data in [('LDA', lda_data), ('SVM', svm_data)]:
        print(f"\n{'='*40}")
        print(f"  {name} BENCHMARK")
        print(f"{'='*40}")
        
        # Sample 32 random points
        n_total = len(data)
        indices = np.random.choice(n_total, 32, replace=False)
        
        X = data[indices, :3]  # First 3 columns are hyperparameters
        y = data[indices, 3]   # 4th column is objective
        
        print(f"\nSampled {len(X)} points from {n_total} total")
        print(f"X shape: {X.shape}, y range: [{y.min():.4f}, {y.max():.4f}]")
        
        # Try with log transformation
        y_log = np.log(y + 1)
        print(f"Log(y+1) range: [{y_log.min():.4f}, {y_log.max():.4f}]")
        
        # Search kernels
        print("\n--- Model Search (log-transformed) ---")
        results = search_kernels(X, y_log, noise_level=0.001, verbose=True)
        
        print(f"\n*** Best Model for {name}: {results[0]['name']} ***")
        print(f"    BIC: {results[0]['bic']:.4f}")
        
        results_all[name] = results
    
    return results_all


def main():
    """Run all model fitting analyses."""
    print("\n" + "=" * 70)
    print("MODEL FITTING FOR BAYESIAN OPTIMIZATION PROJECT")
    print("=" * 70)
    
    # Output directory
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Part 1: Original Branin
    X_train, y_train, gp_original = analyze_branin_original(save_dir)
    
    # Part 1b: High noise analysis (for comparison)
    gp_high_noise = analyze_high_noise(X_train, y_train, save_dir)
    
    # Part 2: Log-transformed Branin  
    gp_log, y_log = analyze_branin_log_transformed(X_train, save_dir)
    
    # Part 3: BIC and model search
    branin_results = compute_bic_analysis(gp_log, y_log, X_train, save_dir)

    # Extra: model search on original (untransformed) Branin scale
    print("\n--- Searching Over Different Kernels (original Branin scale) ---")
    branin_original_results = search_kernels(X_train, y_train, noise_level=0.001)
    best_orig = branin_original_results[0]
    print(f"\n*** Best Original-Scale Model: {best_orig['name']} ***")
    print(f"    BIC: {best_orig['bic']:.4f}")
    print(f"    Fitted kernel: {best_orig['kernel']}")
    
    # Part 4: Real benchmarks
    benchmark_results = analyze_real_benchmarks(save_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nFiles generated:")
    print("  - baseline_branin_se_posterior_heatmaps.png")
    print("  - explorations/exp_high_noise_posterior_heatmaps.png")
    print("  - baseline_branin_se_zscore.png")
    print("  - baseline_branin_log_se_posterior_heatmaps.png")
    print("  - baseline_branin_log_se_zscore.png")
    
    print("\nBest models found:")
    print(f"  Branin (log): {branin_results[0]['name']} (BIC: {branin_results[0]['bic']:.2f})")
    print(f"  Branin (original): {best_orig['name']} (BIC: {best_orig['bic']:.2f})")
    print(f"  LDA: {benchmark_results['LDA'][0]['name']} (BIC: {benchmark_results['LDA'][0]['bic']:.2f})")
    print(f"  SVM: {benchmark_results['SVM'][0]['name']} (BIC: {benchmark_results['SVM'][0]['bic']:.2f})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
