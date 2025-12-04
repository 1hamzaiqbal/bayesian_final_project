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
    RBF, ConstantKernel, Matern, RationalQuadratic, 
    WhiteKernel, ExpSineSquared, DotProduct
)

# Add parent directory to path to import branin function
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def fit_gp(X, y, kernel, noise_level=0.001, n_restarts=10):
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
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise_level**2,  # alpha is variance, not std
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=42
    )
    gp.fit(X, y)
    return gp


def compute_bic(gp, X, y):
    """
    Compute Bayesian Information Criterion.
    
    BIC = k * log(n) - 2 * log_likelihood
    
    where k = number of hyperparameters, n = number of observations
    """
    log_likelihood = gp.log_marginal_likelihood_value_
    n = len(y)
    
    # Count hyperparameters from kernel
    k = gp.kernel_.n_dims
    
    bic = k * np.log(n) - 2 * log_likelihood
    return bic, log_likelihood, k


def create_posterior_heatmaps(gp, X_train, y_train, bounds, title_prefix="", save_dir=None):
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
        filename = f"{title_prefix.lower().replace(' ', '_')}posterior_heatmaps.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.close()
    
    return y_mean, y_std, y_true


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
    """Plot KDE of z-scores compared to standard normal."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
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
    
    # Add statistics
    mean_z = np.mean(z_filtered)
    std_z = np.std(z_filtered)
    ax.text(0.95, 0.95, f'Mean: {mean_z:.3f}\nStd: {std_z:.3f}', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.basename(save_path)}")
    
    plt.close()
    
    return mean_z, std_z


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
    
    results = []
    
    for name, kernel in kernels.items():
        try:
            gp = fit_gp(X, y, kernel, noise_level=noise_level)
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
    
    # Fit GP with SE kernel
    print("\n--- Fitting GP with Constant Mean + SE Kernel ---")
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
    gp = fit_gp(X_train, y_train, kernel, noise_level=0.001)
    
    # Report hyperparameters
    print("\nLearned Hyperparameters:")
    print(f"  Kernel: {gp.kernel_}")
    print(f"  Log-marginal-likelihood: {gp.log_marginal_likelihood_value_:.4f}")
    
    # Extract specific values
    kernel_params = gp.kernel_.get_params()
    print(f"\n  Interpretation:")
    print(f"    - The output scale (constant kernel) controls the overall variance")
    print(f"    - The length scales control how quickly correlation decays with distance")
    print(f"    - Given domain size ~15 in each dimension, length scales should be O(1-10)")
    
    # Create heatmaps
    print("\n--- Creating Posterior Heatmaps ---")
    y_mean, y_std, y_true = create_posterior_heatmaps(
        gp, X_train, y_train, bounds, 
        title_prefix="Original ", 
        save_dir=save_dir
    )
    
    # Check posterior std at training points
    y_pred_train, y_std_train = gp.predict(X_train, return_std=True)
    print(f"\nPosterior std at training points:")
    print(f"  Min: {y_std_train.min():.6f}")
    print(f"  Max: {y_std_train.max():.6f}")
    print(f"  Mean: {y_std_train.mean():.6f}")
    print(f"  → Should be near zero (we set noise ≈ 0.001)")
    
    # Z-score calibration
    print("\n--- Z-Score Calibration Analysis ---")
    z_scores = compute_zscores(gp, X_train, use_log_transform=False)
    mean_z, std_z = plot_zscore_kde(
        z_scores, 
        "Z-Scores: Original Branin Data",
        save_path=os.path.join(save_dir, "zscore_original.png")
    )
    print(f"  Z-score mean: {mean_z:.3f} (should be ≈ 0)")
    print(f"  Z-score std: {std_z:.3f} (should be ≈ 1)")
    
    calibration_quality = "well" if (abs(mean_z) < 0.5 and 0.5 < std_z < 2.0) else "poorly"
    print(f"  → Model appears {calibration_quality} calibrated")
    
    return X_train, y_train, gp


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
    gp_log = fit_gp(X_train, y_log, kernel, noise_level=0.001)
    
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
    plt.savefig(os.path.join(save_dir, "log_transformed_posterior_heatmaps.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: log_transformed_posterior_heatmaps.png")
    
    # Z-score calibration
    print("\n--- Z-Score Calibration (Log-Transformed) ---")
    z_scores = compute_zscores(gp_log, X_train, use_log_transform=True)
    mean_z, std_z = plot_zscore_kde(
        z_scores,
        "Z-Scores: Log-Transformed Branin Data",
        save_path=os.path.join(save_dir, "zscore_log_transformed.png")
    )
    print(f"  Z-score mean: {mean_z:.3f}")
    print(f"  Z-score std: {std_z:.3f}")
    
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
    
    # Part 2: Log-transformed Branin  
    gp_log, y_log = analyze_branin_log_transformed(X_train, save_dir)
    
    # Part 3: BIC and model search
    branin_results = compute_bic_analysis(gp_log, y_log, X_train, save_dir)
    
    # Part 4: Real benchmarks
    benchmark_results = analyze_real_benchmarks(save_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nFiles generated:")
    print("  - original_posterior_heatmaps.png")
    print("  - zscore_original.png")
    print("  - log_transformed_posterior_heatmaps.png")
    print("  - zscore_log_transformed.png")
    
    print("\nBest models found:")
    print(f"  Branin (log): {branin_results[0]['name']} (BIC: {branin_results[0]['bic']:.2f})")
    print(f"  LDA: {benchmark_results['LDA'][0]['name']} (BIC: {benchmark_results['LDA'][0]['bic']:.2f})")
    print(f"  SVM: {benchmark_results['SVM'][0]['name']} (BIC: {benchmark_results['SVM'][0]['bic']:.2f})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
