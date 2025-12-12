"""
Bayesian Optimization for Hyperparameter Tuning

This script implements all bullet points from the Bayesian optimization section:
1. Expected Improvement acquisition function (Snoek et al.)
2. Heatmaps for posterior mean, std, and EI
3. BO experiments on Branin, SVM, LDA
4. Gap metric evaluation
5. Comparison with random search
6. Learning curves and speedup analysis

Author: Generated for Bayesian Optimization final project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import qmc, norm
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, Matern
)


def branin(x1, x2, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    """Branin function (minimization)."""
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    return term1 + term2 + s


def expected_improvement(mu, sigma, f_best, xi=0.01):
    """
    Expected Improvement acquisition function for MINIMIZATION.
    
    From Snoek et al. (2012), Equation (2):
    EI(x) = σ(x) * (γ(x) * Φ(γ(x)) + φ(γ(x)))
    
    where γ(x) = (f_best - μ(x)) / σ(x) for minimization
    
    Parameters:
    -----------
    mu : array
        Posterior mean predictions
    sigma : array
        Posterior standard deviation predictions
    f_best : float
        Best (minimum) function value observed so far
    xi : float
        Exploration-exploitation trade-off parameter
        
    Returns:
    --------
    ei : array
        Expected improvement values
    """
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # For minimization: improvement is f_best - f(x), so γ = (f_best - μ) / σ
    gamma = (f_best - mu - xi) / sigma
    
    # EI = σ * (γ * Φ(γ) + φ(γ))
    # where Φ is the CDF and φ is the PDF of standard normal
    ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
    
    # EI should be non-negative
    ei = np.maximum(ei, 0)
    
    return ei


def fit_gp(X, y, kernel=None, noise_level=0.001):
    """Fit a Gaussian process model."""
    if kernel is None:
        n_dim = X.shape[1]
        kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0]*n_dim)
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise_level**2,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42
    )
    gp.fit(X, y)
    return gp


def compute_gap(f_best_found, f_best_initial, f_optimum):
    """
    Compute the gap metric for minimization.
    
    For minimization, a lower value is better, so:
    gap = (f_best_initial - f_best_found) / (f_best_initial - f_optimum)
    
    This gives 0 if we haven't improved, 1 if we found the optimum.
    """
    denominator = f_best_initial - f_optimum
    if abs(denominator) < 1e-10:
        return 1.0 if f_best_found <= f_optimum else 0.0
    
    gap = (f_best_initial - f_best_found) / denominator
    return np.clip(gap, 0, 1)


def bayesian_optimization(X_pool, y_pool, n_initial=5, n_iterations=30, 
                          use_log_transform=False, init_indices=None, 
                          kernel=None, verbose=False):
    """
    Run Bayesian optimization experiment.
    
    Parameters:
    -----------
    X_pool : array, shape (n, d)
        Pool of candidate points
    y_pool : array, shape (n,)
        Function values at candidate points
    n_initial : int
        Number of initial random points
    n_iterations : int
        Number of BO iterations
    use_log_transform : bool
        Whether to log-transform the objective
    init_indices : array, optional
        Pre-specified initial indices for reproducibility (for paired comparison)
    kernel : sklearn kernel, optional
        GP kernel to use (default: RBF)
    verbose : bool
        Print progress
        
    Returns:
    --------
    history : dict
        Contains indices selected, y values, and best values at each step
    """
    n_pool = len(X_pool)
    available = np.ones(n_pool, dtype=bool)
    
    # Store history
    selected_indices = []
    y_values = []
    best_so_far = []
    
    # Use provided init_indices OR generate random ones
    if init_indices is None:
        rng = np.random.default_rng()
        init_indices = rng.choice(n_pool, size=n_initial, replace=False)
    
    for idx in init_indices:
        selected_indices.append(idx)
        y_values.append(y_pool[idx])
        available[idx] = False
        best_so_far.append(min(y_values))
    
    # BO loop
    for iteration in range(n_iterations):
        # Current data
        X_train = X_pool[selected_indices]
        y_train = np.array(y_values)
        
        if use_log_transform:
            y_train_transformed = np.log(y_train + 1)
        else:
            y_train_transformed = y_train
        
        # Fit GP with specified kernel
        gp = fit_gp(X_train, y_train_transformed, kernel=kernel)
        
        # Predict on available points
        available_indices = np.where(available)[0]
        X_available = X_pool[available_indices]
        
        mu, sigma = gp.predict(X_available, return_std=True)
        
        if use_log_transform:
            f_best = np.log(min(y_values) + 1)
        else:
            f_best = min(y_values)
        
        # Compute EI
        ei = expected_improvement(mu, sigma, f_best)
        
        # Select point with maximum EI
        best_idx_local = np.argmax(ei)
        best_idx_global = available_indices[best_idx_local]
        
        # Add to dataset
        selected_indices.append(best_idx_global)
        y_values.append(y_pool[best_idx_global])
        available[best_idx_global] = False
        best_so_far.append(min(y_values))
        
        if verbose and (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}: best = {min(y_values):.4f}")
    
    return {
        'indices': selected_indices,
        'y_values': y_values,
        'best_so_far': best_so_far,
        'init_indices': init_indices  # Return for verification
    }


def random_search(X_pool, y_pool, n_initial=5, n_iterations=30, 
                  init_indices=None, rng=None, verbose=False):
    """
    Run random search experiment.
    
    Parameters:
    -----------
    init_indices : array, optional
        Pre-specified initial indices for reproducibility (for paired comparison)
    rng : numpy Generator, optional
        Random number generator for reproducibility
    """
    n_pool = len(X_pool)
    available = np.ones(n_pool, dtype=bool)
    
    selected_indices = []
    y_values = []
    best_so_far = []
    
    # Use provided init_indices OR generate random ones
    if rng is None:
        rng = np.random.default_rng()
    
    if init_indices is None:
        init_indices = rng.choice(n_pool, size=n_initial, replace=False)
    
    for idx in init_indices:
        selected_indices.append(idx)
        y_values.append(y_pool[idx])
        available[idx] = False
        best_so_far.append(min(y_values))
    
    # Random selection
    for iteration in range(n_iterations):
        available_indices = np.where(available)[0]
        if len(available_indices) == 0:
            break
            
        idx = rng.choice(available_indices)
        selected_indices.append(idx)
        y_values.append(y_pool[idx])
        available[idx] = False
        best_so_far.append(min(y_values))
    
    return {
        'indices': selected_indices,
        'y_values': y_values,
        'best_so_far': best_so_far,
        'init_indices': init_indices  # Return for verification
    }


def create_ei_heatmaps(save_dir):
    """
    Create heatmaps for posterior mean, std, and EI for Branin.
    """
    print("=" * 70)
    print("BRANIN FUNCTION: EI HEATMAPS")
    print("=" * 70)
    
    # Generate Sobol training points (same as before)
    bounds = [(-5, 10), (0, 15)]
    sampler = qmc.Sobol(d=2, scramble=True, seed=42)
    X_train = sampler.random(32)
    X_train = qmc.scale(X_train, [-5, 0], [10, 15])
    y_train = branin(X_train[:, 0], X_train[:, 1])
    
    # Log transform
    y_train_log = np.log(y_train + 1)
    
    # Fit GP with best model from model fitting (SE kernel)
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0])
    gp = fit_gp(X_train, y_train_log, kernel)
    
    print(f"Fitted GP on 32 log-transformed Branin points")
    print(f"Kernel: {gp.kernel_}")
    
    # Create prediction grid
    x1 = np.linspace(-5, 10, 100)
    x2 = np.linspace(0, 15, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Get predictions
    mu, sigma = gp.predict(X_grid, return_std=True)
    
    # Compute EI
    f_best = y_train_log.min()
    ei = expected_improvement(mu, sigma, f_best)
    
    mu = mu.reshape(X1.shape)
    sigma = sigma.reshape(X1.shape)
    ei = ei.reshape(X1.shape)
    
    # Find EI maximum
    max_idx = np.unravel_index(np.argmax(ei), ei.shape)
    max_x1 = x1[max_idx[1]]
    max_x2 = x2[max_idx[0]]
    
    print(f"\nEI maximum at: x1={max_x1:.3f}, x2={max_x2:.3f}")
    print(f"EI value at maximum: {ei[max_idx]:.6f}")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Posterior mean
    im0 = axes[0].imshow(mu, extent=[-5, 10, 0, 15], origin='lower', aspect='auto', cmap='viridis')
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c='red', s=30, edgecolor='white', zorder=5)
    axes[0].set_xlabel('$x_1$'); axes[0].set_ylabel('$x_2$')
    axes[0].set_title('GP Posterior Mean (log scale)')
    plt.colorbar(im0, ax=axes[0], label='$\\mu(x)$')
    
    # Posterior std
    im1 = axes[1].imshow(sigma, extent=[-5, 10, 0, 15], origin='lower', aspect='auto', cmap='plasma')
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c='white', s=30, edgecolor='black', zorder=5)
    axes[1].set_xlabel('$x_1$'); axes[1].set_ylabel('$x_2$')
    axes[1].set_title('GP Posterior Std Dev')
    plt.colorbar(im1, ax=axes[1], label='$\\sigma(x)$')
    
    # EI
    im2 = axes[2].imshow(ei, extent=[-5, 10, 0, 15], origin='lower', aspect='auto', cmap='hot')
    axes[2].scatter(X_train[:, 0], X_train[:, 1], c='white', s=30, edgecolor='black', zorder=5)
    axes[2].plot(max_x1, max_x2, 'c*', markersize=20, markeredgecolor='white', 
                 markeredgewidth=2, label=f'EI max ({max_x1:.2f}, {max_x2:.2f})')
    axes[2].set_xlabel('$x_1$'); axes[2].set_ylabel('$x_2$')
    axes[2].set_title('Expected Improvement')
    plt.colorbar(im2, ax=axes[2], label='EI(x)')
    axes[2].legend(loc='upper right')
    
    plt.suptitle('Branin Function: GP Posterior and Expected Improvement', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ei_heatmaps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: ei_heatmaps.png")
    
    # Interpretation
    print("\nInterpretation:")
    print("  The EI maximum is located where uncertainty is high AND predicted value is good.")
    print("  This balances exploration (high σ) and exploitation (low μ).")
    
    return max_x1, max_x2


def run_experiments(n_runs=20, save_dir=None):
    """
    Run 20 BO experiments on all datasets with random search baseline.
    
    FIXED: BO and RS now share identical initial points per run for proper pairing.
    FIXED: LDA/SVM use Matern 3/2 kernel as stated in report.
    """
    print("\n" + "=" * 70)
    print("RUNNING BO AND RANDOM SEARCH EXPERIMENTS")
    print("=" * 70)
    
    # Load datasets
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lda_data = np.loadtxt(os.path.join(parent_dir, 'lda.csv'), delimiter=',')
    svm_data = np.loadtxt(os.path.join(parent_dir, 'svm.csv'), delimiter=',')
    
    # Prepare Branin pool (dense Sobol grid)
    sampler = qmc.Sobol(d=2, scramble=True, seed=123)
    X_branin = sampler.random(1000)
    X_branin = qmc.scale(X_branin, [-5, 0], [10, 15])
    y_branin = branin(X_branin[:, 0], X_branin[:, 1])
    f_opt_branin = 0.397887  # Known optimum
    
    # Branin uses SE kernel, LDA/SVM use Matern 3/2 (as per model fitting results)
    kernel_branin = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0])
    kernel_matern = ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0, 1.0], nu=1.5)
    
    datasets = {
        'Branin': (X_branin, y_branin, f_opt_branin, True, kernel_branin),  # (X, y, f_opt, use_log, kernel)
        'LDA': (lda_data[:, :3], lda_data[:, 3], lda_data[:, 3].min(), True, kernel_matern),
        'SVM': (svm_data[:, :3], svm_data[:, 3], svm_data[:, 3].min(), True, kernel_matern)
    }
    
    results = {}
    
    for name, (X_pool, y_pool, f_opt, use_log, kernel) in datasets.items():
        print(f"\n{'='*50}")
        print(f"  {name} DATASET")
        print(f"{'='*50}")
        print(f"Pool size: {len(X_pool)}, Optimum: {f_opt:.6f}")
        print(f"Kernel: {'RBF' if name == 'Branin' else 'Matern 3/2'}")
        
        bo_results = []
        rs_results = []
        
        for run in range(n_runs):
            # Create seeded RNG for this run - used for BOTH BO and RS
            run_rng = np.random.default_rng(seed=run * 42 + 1)
            
            # Generate SHARED initial indices for proper pairing
            n_pool = len(X_pool)
            init_indices = run_rng.choice(n_pool, size=5, replace=False)
            
            # Create separate RNG for RS post-init (BO uses its own EI-based selection)
            rs_rng = np.random.default_rng(seed=run * 42 + 2)
            
            # Bayesian optimization (5 initial + 30 iterations = 35 total)
            bo_history = bayesian_optimization(
                X_pool, y_pool, n_initial=5, n_iterations=30, 
                use_log_transform=use_log, 
                init_indices=init_indices.copy(),  # Pass shared init
                kernel=kernel,
                verbose=False
            )
            
            # Random search (5 initial + 145 more = 150 total)
            rs_history = random_search(
                X_pool, y_pool, n_initial=5, n_iterations=145, 
                init_indices=init_indices.copy(),  # Same shared init
                rng=rs_rng,
                verbose=False
            )
            
            # Verify pairing
            assert np.array_equal(bo_history['init_indices'], rs_history['init_indices']), \
                f"Run {run}: Init indices don't match!"
            
            bo_results.append(bo_history)
            rs_results.append(rs_history)
            
            if (run + 1) % 5 == 0:
                print(f"  Run {run + 1}/{n_runs} complete")
        
        results[name] = {
            'bo': bo_results,
            'rs': rs_results,
            'f_opt': f_opt
        }
    
    print("\n✓ All runs verified: BO and RS share identical initial points per run")
    return results


def plot_learning_curves(results, save_dir):
    """Create learning curve plots for each dataset."""
    print("\n" + "=" * 70)
    print("CREATING LEARNING CURVE PLOTS")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        f_opt = data['f_opt']
        
        # Compute gaps for BO
        n_obs_bo = len(data['bo'][0]['best_so_far'])
        bo_gaps = np.zeros((len(data['bo']), n_obs_bo))
        
        for run_idx, run in enumerate(data['bo']):
            f_best_initial = run['best_so_far'][4]  # After 5 initial points
            for i, best in enumerate(run['best_so_far']):
                bo_gaps[run_idx, i] = compute_gap(best, f_best_initial, f_opt)
        
        # Compute gaps for RS (first 35 observations for comparison)
        n_obs_rs = 35
        rs_gaps = np.zeros((len(data['rs']), n_obs_rs))
        
        for run_idx, run in enumerate(data['rs']):
            f_best_initial = run['best_so_far'][4]
            for i in range(min(n_obs_rs, len(run['best_so_far']))):
                rs_gaps[run_idx, i] = compute_gap(run['best_so_far'][i], f_best_initial, f_opt)
        
        # Use Standard Error bands (not std, to show uncertainty in mean)
        x_bo = np.arange(1, n_obs_bo + 1)
        x_rs = np.arange(1, n_obs_rs + 1)
        n_runs = bo_gaps.shape[0]
        bo_se = bo_gaps.std(axis=0) / np.sqrt(n_runs)
        rs_se = rs_gaps.std(axis=0) / np.sqrt(n_runs)
        
        ax.plot(x_bo, bo_gaps.mean(axis=0), 'b-', linewidth=2, label='BO (EI)')
        ax.fill_between(x_bo, 
                        np.clip(bo_gaps.mean(axis=0) - bo_se, 0, 1),
                        np.clip(bo_gaps.mean(axis=0) + bo_se, 0, 1),
                        alpha=0.3, color='b')
        
        ax.plot(x_rs, rs_gaps.mean(axis=0), 'r--', linewidth=2, label='Random Search')
        ax.fill_between(x_rs,
                        np.clip(rs_gaps.mean(axis=0) - rs_se, 0, 1),
                        np.clip(rs_gaps.mean(axis=0) + rs_se, 0, 1),
                        alpha=0.3, color='r')
        
        ax.set_xlabel('Number of Observations')
        ax.set_ylabel('Gap (higher is better)')
        ax.set_title(f'{name} Dataset')
        ax.legend()
        ax.set_xlim([1, 35])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Learning Curves: BO vs Random Search', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: learning_curves.png")


def compute_statistics(results, save_dir):
    """Compute gap statistics and perform t-tests."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    stats_results = {}
    
    for name, data in results.items():
        print(f"\n{'='*50}")
        print(f"  {name} DATASET")
        print(f"{'='*50}")
        
        f_opt = data['f_opt']
        
        # Final gaps for BO (at 35 observations)
        bo_final_gaps = []
        for run in data['bo']:
            f_best_initial = run['best_so_far'][4]
            f_best_found = run['best_so_far'][-1]
            gap = compute_gap(f_best_found, f_best_initial, f_opt)
            bo_final_gaps.append(gap)
        
        bo_final_gaps = np.array(bo_final_gaps)
        
        # Gaps for RS at different observation counts
        rs_gaps_at = {}
        for n_obs in [30, 60, 90, 120, 150]:
            gaps = []
            for run in data['rs']:
                if len(run['best_so_far']) >= n_obs:
                    f_best_initial = run['best_so_far'][4]
                    f_best_found = run['best_so_far'][n_obs - 1]
                    gap = compute_gap(f_best_found, f_best_initial, f_opt)
                    gaps.append(gap)
            rs_gaps_at[n_obs] = np.array(gaps)
        
        # Print results - FIXED: BO has 35 total evals, not 30
        print(f"\nMean Gap (±std):")
        print(f"  BO (35 total evals):  {bo_final_gaps.mean():.4f} ± {bo_final_gaps.std():.4f}")
        
        for n_obs in [30, 60, 90, 120, 150]:
            gaps = rs_gaps_at[n_obs]
            print(f"  RS ({n_obs:3d} total evals): {gaps.mean():.4f} ± {gaps.std():.4f}")
        
        # Paired t-test: BO@35 vs RS@N
        print(f"\nPaired t-tests (BO@35 vs RS@N):")
        print(f"  (Note: p > 0.05 means 'no significant difference', not 'equivalence')")
        
        speedup = None
        for n_obs in [30, 60, 90, 120, 150]:
            if len(rs_gaps_at[n_obs]) == len(bo_final_gaps):
                t_stat, p_value = stats.ttest_rel(bo_final_gaps, rs_gaps_at[n_obs])
                mean_diff = bo_final_gaps.mean() - rs_gaps_at[n_obs].mean()
                print(f"  RS@{n_obs:3d}: t={t_stat:7.3f}, p={p_value:.4f}, ΔGap={mean_diff:+.3f}", end="")
                if p_value < 0.05:
                    print(" *")
                else:
                    print(" (n.s.)")
                    if speedup is None:
                        speedup = n_obs
        
        if speedup:
            print(f"\n  At RS@{speedup}, the BO-RS difference is no longer statistically significant (p>0.05).")
            print(f"  However, this does NOT prove equivalence—only inconclusive difference.")
        
        stats_results[name] = {
            'bo_mean': bo_final_gaps.mean(),
            'bo_std': bo_final_gaps.std(),
            'rs_gaps': rs_gaps_at,
            'bo_gaps': bo_final_gaps
        }
    
    return stats_results


def main():
    """Run all Bayesian optimization analyses."""
    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMIZATION FOR HYPERPARAMETER TUNING")
    print("=" * 70)
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create EI heatmaps for Branin
    max_x1, max_x2 = create_ei_heatmaps(save_dir)
    
    # Run experiments
    results = run_experiments(n_runs=20, save_dir=save_dir)
    
    # Plot learning curves
    plot_learning_curves(results, save_dir)
    
    # Compute statistics
    stats_results = compute_statistics(results, save_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nFiles generated:")
    print("  - ei_heatmaps.png")
    print("  - learning_curves.png")
    
    print("\nKey findings:")
    for name, stats in stats_results.items():
        print(f"  {name}: BO gap = {stats['bo_mean']:.3f} vs RS@30 = {stats['rs_gaps'][30].mean():.3f}")


if __name__ == "__main__":
    main()
