"""
Bonus: Acquisition Function Comparison

This script implements and compares multiple acquisition functions:
1. Expected Improvement (EI) - Already implemented
2. Probability of Improvement (PI) - Kushner (1964)
3. GP Lower Confidence Bound (LCB) - Srinivas et al. (2010)

We investigate which acquisition function performs best on different
problem types and how the choice of κ in UCB affects performance.

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
from sklearn.gaussian_process.kernels import Kernel, RBF, ConstantKernel, Matern, ExpSineSquared


def branin(x1, x2, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    """Branin function (minimization)."""
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    return term1 + term2 + s


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


def expected_improvement(mu, sigma, f_best, xi=0.01):
    """
    Expected Improvement (EI) for minimization.
    
    EI(x) = σ * (γ * Φ(γ) + φ(γ))
    where γ = (f_best - μ - ξ) / σ
    """
    sigma = np.maximum(sigma, 1e-9)
    gamma = (f_best - mu - xi) / sigma
    ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
    return np.maximum(ei, 0)


def probability_of_improvement(mu, sigma, f_best, xi=0.01):
    """
    Probability of Improvement (PI) for minimization.
    
    From Kushner (1964):
    PI(x) = Φ(γ)
    where γ = (f_best - μ - ξ) / σ
    """
    sigma = np.maximum(sigma, 1e-9)
    gamma = (f_best - mu - xi) / sigma
    return norm.cdf(gamma)


def lower_confidence_bound(mu, sigma, kappa=2.0):
    """
    GP Lower Confidence Bound (LCB) for minimization.
    
    From Srinivas et al. (2010):
    LCB(x) = μ(x) - κ * σ(x)
    
    We want to MINIMIZE LCB to find good points.
    For acquisition, we return -LCB so we can MAXIMIZE.
    """
    return -(mu - kappa * sigma)  # Negative so we maximize


def fit_gp(X, y, kernel=None, noise_level=0.001, normalize_y=True, n_restarts=8):
    """Fit a Gaussian process model, respecting noise in original units."""
    if kernel is None:
        n_dim = X.shape[1]
        kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0]*n_dim)
    
    # sklearn standardizes y internally when normalize_y=True; scale alpha so
    # noise_level is interpreted in original output units.
    alpha = noise_level**2
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


def compute_gap(f_best_found, f_best_initial, f_optimum):
    """Compute gap metric for minimization."""
    denominator = f_best_initial - f_optimum
    if abs(denominator) < 1e-10:
        return 1.0 if f_best_found <= f_optimum else 0.0
    gap = (f_best_initial - f_best_found) / denominator
    return np.clip(gap, 0, 1)


def random_search(X_pool, y_pool, n_initial=5, n_iterations=30, init_indices=None, rng=None):
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
    
    if rng is None:
        rng = np.random.default_rng()
    
    if init_indices is None:
        init_indices = rng.choice(n_pool, size=n_initial, replace=False)
    
    for idx in init_indices:
        selected_indices.append(idx)
        y_values.append(y_pool[idx])
        available[idx] = False
        best_so_far.append(min(y_values))
    
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
        'init_indices': init_indices
    }


def bayesian_optimization_with_acq(X_pool, y_pool, acquisition_fn, acq_params,
                                    n_initial=5, n_iterations=30, 
                                    use_log_transform=False, init_indices=None,
                                    kernel=None):
    """
    Run Bayesian optimization with a specified acquisition function.
    
    Parameters:
    -----------
    init_indices : array, optional
        Pre-specified initial indices for reproducibility (for paired comparison)
    """
    n_pool = len(X_pool)
    available = np.ones(n_pool, dtype=bool)
    
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
    
    for iteration in range(n_iterations):
        X_train = X_pool[selected_indices]
        y_train = np.array(y_values)
        
        if use_log_transform:
            y_train_transformed = np.log(y_train + 1)
        else:
            y_train_transformed = y_train
        
        gp = fit_gp(X_train, y_train_transformed, kernel=kernel)
        
        available_indices = np.where(available)[0]
        X_available = X_pool[available_indices]
        
        mu, sigma = gp.predict(X_available, return_std=True)
        
        if use_log_transform:
            f_best = np.log(min(y_values) + 1)
        else:
            f_best = min(y_values)
        
        # Compute acquisition values
        if acquisition_fn == 'EI':
            acq_values = expected_improvement(mu, sigma, f_best, xi=acq_params.get('xi', 0.01))
        elif acquisition_fn == 'PI':
            acq_values = probability_of_improvement(mu, sigma, f_best, xi=acq_params.get('xi', 0.01))
        elif acquisition_fn == 'LCB':
            acq_values = lower_confidence_bound(mu, sigma, kappa=acq_params.get('kappa', 2.0))
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_fn}")
        
        best_idx_local = np.argmax(acq_values)
        best_idx_global = available_indices[best_idx_local]
        
        selected_indices.append(best_idx_global)
        y_values.append(y_pool[best_idx_global])
        available[best_idx_global] = False
        best_so_far.append(min(y_values))
    
    return {
        'indices': selected_indices,
        'y_values': y_values,
        'best_so_far': best_so_far,
        'init_indices': init_indices  # Return for verification
    }


def run_acquisition_comparison(n_runs=20, save_dir=None):
    """
    Compare EI, PI, LCB, and Random Search acquisition functions.
    
    FIXED: All methods share identical initial points per run for proper paired comparison.
    ADDED: Random search baseline for calibration.
    """
    print("=" * 70)
    print("ACQUISITION FUNCTION COMPARISON")
    print("=" * 70)
    print("NOTE: All methods share identical init points per run (proper pairing)")
    print("NOTE: Gap computed on original y; GP may model transformed y per dataset")
    
    # Load datasets
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lda_data = np.loadtxt(os.path.join(parent_dir, 'lda.csv'), delimiter=',')
    svm_data = np.loadtxt(os.path.join(parent_dir, 'svm.csv'), delimiter=',')
    
    # Branin pool
    sampler = qmc.Sobol(d=2, scramble=True, seed=123)
    X_branin = sampler.random(1000)
    X_branin = qmc.scale(X_branin, [-5, 0], [10, 15])
    y_branin = branin(X_branin[:, 0], X_branin[:, 1])
    f_opt_branin = 0.397887
    
    # Improved modeling choices:
    # - Branin: original scale + SE + Periodic(x1)
    # - LDA/SVM: log(y+1) + Matern 3/2
    rbf2 = RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
    periodic_x1 = ActiveDimKernel(ExpSineSquared(length_scale=1.0, periodicity=2 * np.pi), active_dims=[0])
    kernel_branin = ConstantKernel(1.0, (1e-3, 1e3)) * (rbf2 + periodic_x1)
    kernel_matern = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0, 1.0, 1.0],
        length_scale_bounds=(1e-2, 1e2),
        nu=1.5,
    )

    datasets = {
        'Branin': (X_branin, y_branin, f_opt_branin, False, kernel_branin),
        'LDA': (lda_data[:, :3], lda_data[:, 3], lda_data[:, 3].min(), True, kernel_matern),
        'SVM': (svm_data[:, :3], svm_data[:, 3], svm_data[:, 3].min(), True, kernel_matern)
    }
    
    # Acquisition functions to compare (including Random Search as baseline)
    acquisition_configs = {
        'Random': {},  # Random search baseline
        'EI (ξ=0.01)': {'xi': 0.01},
        'PI (ξ=0.01)': {'xi': 0.01},
        'LCB (κ=1)': {'kappa': 1.0},
        'LCB (κ=2)': {'kappa': 2.0},
    }
    
    all_results = {}
    
    for name, (X_pool, y_pool, f_opt, use_log, kernel) in datasets.items():
        print(f"\n{'='*50}")
        print(f"  {name} DATASET")
        print(f"{'='*50}")
        
        n_pool = len(X_pool)
        dataset_results = {}
        
        # Pre-generate all init_indices for this dataset (shared across methods)
        all_init_indices = []
        for run in range(n_runs):
            run_rng = np.random.default_rng(seed=run * 42 + 1)
            init_indices = run_rng.choice(n_pool, size=5, replace=False)
            all_init_indices.append(init_indices)
        
        for acq_name, acq_params in acquisition_configs.items():
            print(f"\n  Testing {acq_name}...")
            
            runs = []
            for run in range(n_runs):
                init_indices = all_init_indices[run].copy()
                
                if acq_name == 'Random':
                    # Random search uses same init but random subsequent selections
                    rs_rng = np.random.default_rng(seed=run * 42 + 100)
                    history = random_search(
                        X_pool, y_pool, n_initial=5, n_iterations=30,
                        init_indices=init_indices, rng=rs_rng
                    )
                else:
                    # Determine base acquisition function
                    if acq_name.startswith('LCB'):
                        acq_fn = 'LCB'
                    elif acq_name.startswith('EI'):
                        acq_fn = 'EI'
                    elif acq_name.startswith('PI'):
                        acq_fn = 'PI'
                    else:
                        acq_fn = acq_name
                    
                    history = bayesian_optimization_with_acq(
                        X_pool, y_pool, acq_fn, acq_params,
                        n_initial=5, n_iterations=30,
                        use_log_transform=use_log,
                        init_indices=init_indices,
                        kernel=kernel
                    )
                
                # Verify pairing
                assert np.array_equal(history['init_indices'], all_init_indices[run]), \
                    f"Run {run}: Init indices don't match!"
                
                runs.append(history)
            
            dataset_results[acq_name] = runs
            
            # Compute final gaps
            final_gaps = []
            for run in runs:
                f_best_initial = run['best_so_far'][4]
                f_best_found = run['best_so_far'][-1]
                gap = compute_gap(f_best_found, f_best_initial, f_opt)
                final_gaps.append(gap)
            
            print(f"    Mean gap: {np.mean(final_gaps):.4f} ± {np.std(final_gaps):.4f}")
        
        all_results[name] = {
            'results': dataset_results,
            'f_opt': f_opt
        }
    
    print("\n✓ All runs verified: All methods share identical initial points per run")
    return all_results


def plot_comparison_results(all_results, save_dir):
    """Create comparison plots for acquisition functions."""
    print("\n" + "=" * 70)
    print("CREATING COMPARISON PLOTS")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = {
        'Random': '#7f7f7f',  # Gray for baseline
        'EI (ξ=0.01)': '#1f77b4',
        'PI (ξ=0.01)': '#ff7f0e', 
        'LCB (κ=1)': '#2ca02c',
        'LCB (κ=2)': '#d62728',
    }
    
    for idx, (dataset_name, data) in enumerate(all_results.items()):
        ax = axes[idx]
        f_opt = data['f_opt']
        
        for acq_name, runs in data['results'].items():
            n_obs = len(runs[0]['best_so_far'])
            n_runs = len(runs)
            gaps = np.zeros((n_runs, n_obs))
            
            for run_idx, run in enumerate(runs):
                f_best_initial = run['best_so_far'][4]
                for i, best in enumerate(run['best_so_far']):
                    gaps[run_idx, i] = compute_gap(best, f_best_initial, f_opt)
            
            # Start x-axis at 5 (gap only defined after init points)
            x = np.arange(5, n_obs + 1)
            mean_gap = gaps.mean(axis=0)[4:]  # Skip first 4 (gap undefined/0)
            se_gap = gaps.std(axis=0)[4:] / np.sqrt(n_runs)  # Standard Error
            
            color = colors.get(acq_name, '#000000')
            linestyle = '--' if acq_name == 'Random' else '-'
            ax.plot(x, mean_gap, label=acq_name, color=color, linewidth=2, linestyle=linestyle)
            ax.fill_between(x, 
                           np.clip(mean_gap - se_gap, 0, 1), 
                           np.clip(mean_gap + se_gap, 0, 1), 
                           alpha=0.2, color=color)
        
        ax.set_xlabel('Number of Observations')
        ax.set_ylabel('Gap (higher is better)')
        ax.set_title(f'{dataset_name} Dataset')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim([5, 35])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Acquisition Function Comparison: EI vs PI vs LCB', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acquisition_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: acquisition_comparison.png")


def compute_statistics_and_rankings(all_results, save_dir):
    """Compute statistics and rank acquisition functions."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS AND RANKINGS")
    print("=" * 70)
    print("NOTE: p > 0.05 means 'no significant difference', not 'equivalence'")
    print("NOTE: Multiple comparisons not corrected (interpret with caution)")
    
    summary_stats = {}
    
    for dataset_name, data in all_results.items():
        print(f"\n{'='*50}")
        print(f"  {dataset_name} DATASET")
        print(f"{'='*50}")
        
        f_opt = data['f_opt']
        stats_list = []
        n_runs = None
        
        for acq_name, runs in data['results'].items():
            final_gaps = []
            for run in runs:
                f_best_initial = run['best_so_far'][4]
                f_best_found = run['best_so_far'][-1]
                gap = compute_gap(f_best_found, f_best_initial, f_opt)
                final_gaps.append(gap)
            
            final_gaps = np.array(final_gaps)
            n_runs = len(final_gaps)
            stats_list.append({
                'name': acq_name,
                'mean': final_gaps.mean(),
                'std': final_gaps.std(),
                'se': final_gaps.std() / np.sqrt(n_runs),
                'gaps': final_gaps
            })
        
        # Sort by mean gap (descending)
        stats_list.sort(key=lambda x: x['mean'], reverse=True)
        
        print(f"\n  Rankings (by mean gap, n={n_runs} runs):")
        print(f"  {'Rank':<5} {'Acquisition':<15} {'Mean Gap':<12} {'±SE':<10}")
        print("  " + "-" * 45)
        
        for rank, stat in enumerate(stats_list):
            print(f"  {rank+1:<5} {stat['name']:<15} {stat['mean']:<12.4f} ±{stat['se']:<9.4f}")
        
        # Paired t-tests: compare best against others
        best = stats_list[0]
        print(f"\n  Paired t-tests ({best['name']} vs others):")
        print(f"  (Differences are NOT corrected for multiple comparisons)")
        
        for stat in stats_list[1:]:
            diff = best['gaps'] - stat['gaps']
            if np.allclose(diff, 0):
                t_stat, p_value = 0.0, 1.0
            else:
                t_stat, p_value = stats.ttest_rel(best['gaps'], stat['gaps'])

            pooled_std = np.sqrt((best['std']**2 + stat['std']**2) / 2)
            effect_size = 0.0 if pooled_std == 0 else (best['mean'] - stat['mean']) / pooled_std
            sig_marker = "*" if p_value < 0.05 else "(n.s.)"
            print(f"    vs {stat['name']:<12}: t={t_stat:7.3f}, p={p_value:.4f}, d={effect_size:+.2f} {sig_marker}")
        
        summary_stats[dataset_name] = stats_list
    
    return summary_stats


def create_kappa_sensitivity_plot(save_dir):
    """Investigate sensitivity to κ parameter in LCB."""
    print("\n" + "=" * 70)
    print("LCB KAPPA SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Branin pool
    sampler = qmc.Sobol(d=2, scramble=True, seed=123)
    X_pool = sampler.random(1000)
    X_pool = qmc.scale(X_pool, [-5, 0], [10, 15])
    y_pool = branin(X_pool[:, 0], X_pool[:, 1])
    f_opt = 0.397887

    # Use the improved Branin model: original scale + SE + Periodic(x1)
    rbf2 = RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
    periodic_x1 = ActiveDimKernel(ExpSineSquared(length_scale=1.0, periodicity=2 * np.pi), active_dims=[0])
    kernel_branin = ConstantKernel(1.0, (1e-3, 1e3)) * (rbf2 + periodic_x1)
    
    kappa_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    n_runs = 20
    n_pool = len(X_pool)
    
    # Pre-generate all init_indices (SHARED across all κ values for fair comparison)
    all_init_indices = []
    for run in range(n_runs):
        run_rng = np.random.default_rng(seed=run * 42 + 1)
        init_indices = run_rng.choice(n_pool, size=5, replace=False)
        all_init_indices.append(init_indices)
    
    results_by_kappa = {}
    
    for kappa in kappa_values:
        print(f"  Testing κ = {kappa}...")
        
        gaps = []
        for run in range(n_runs):
            init_indices = all_init_indices[run].copy()
            
            history = bayesian_optimization_with_acq(
                X_pool, y_pool, 'LCB', {'kappa': kappa},
                n_initial=5, n_iterations=30,
                use_log_transform=False,
                init_indices=init_indices,
                kernel=kernel_branin
            )
            
            f_best_initial = history['best_so_far'][4]
            f_best_found = history['best_so_far'][-1]
            gap = compute_gap(f_best_found, f_best_initial, f_opt)
            gaps.append(gap)
        
        results_by_kappa[kappa] = np.array(gaps)
    
    # Plot with SE (not std)
    kappas = list(results_by_kappa.keys())
    means = [results_by_kappa[k].mean() for k in kappas]
    ses = [results_by_kappa[k].std() / np.sqrt(n_runs) for k in kappas]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(kappas, means, yerr=ses, marker='o', capsize=5, 
                linewidth=2, markersize=8, color='#d62728')
    
    ax.set_xlabel('κ (exploration-exploitation trade-off)', fontsize=11)
    ax.set_ylabel('Mean Gap (±SE)', fontsize=11)
    ax.set_title('LCB Sensitivity to κ on Branin Function\n(All κ values share identical init points per run)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])
    
    # Find best κ but check if statistically indistinguishable from others
    best_idx = np.argmax(means)
    best_kappa = kappas[best_idx]
    spread = float(np.max(means) - np.min(means))
    
    # Check if top-2 are statistically different (or if the curve is essentially flat)
    if spread < 1e-3:
        annotation = "Flat sensitivity (all κ similar)"
    else:
        sorted_idx = np.argsort(means)[::-1]
        if len(sorted_idx) >= 2:
            best_gaps = results_by_kappa[kappas[sorted_idx[0]]]
            second_gaps = results_by_kappa[kappas[sorted_idx[1]]]
            _, p_val = stats.ttest_rel(best_gaps, second_gaps)

            if np.isnan(p_val) or p_val > 0.05:
                annotation = f"Top κ values indistinguishable (p={p_val:.2f})"
            else:
                annotation = f"Best: κ={best_kappa}"
        else:
            annotation = f"Best: κ={best_kappa}"
    
    ax.annotate(annotation, 
                xy=(kappas[best_idx], means[best_idx]),
                xytext=(kappas[best_idx] + 0.8, means[best_idx] - 0.08),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kappa_sensitivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: kappa_sensitivity.png")
    
    # Print full table
    print(f"\n  κ Sensitivity Results (n={n_runs} runs, shared init):")
    print(f"  {'κ':<8} {'Mean Gap':<12} {'±SE':<10}")
    print("  " + "-" * 30)
    for k in kappas:
        mean = results_by_kappa[k].mean()
        se = results_by_kappa[k].std() / np.sqrt(n_runs)
        print(f"  {k:<8.1f} {mean:<12.4f} ±{se:<9.4f}")
    
    print(f"\n  Best κ = {best_kappa} with mean gap = {means[best_idx]:.4f}")
    
    return results_by_kappa


def main():
    """Run the complete acquisition function comparison study."""
    print("\n" + "=" * 70)
    print("BONUS: ACQUISITION FUNCTION COMPARISON STUDY")
    print("=" * 70)
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run main comparison
    all_results = run_acquisition_comparison(n_runs=20, save_dir=save_dir)
    
    # Create comparison plots
    plot_comparison_results(all_results, save_dir)
    
    # Statistical analysis
    summary_stats = compute_statistics_and_rankings(all_results, save_dir)
    
    # Kappa sensitivity analysis
    kappa_results = create_kappa_sensitivity_plot(save_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nBest acquisition function by dataset:")
    for dataset, stats in summary_stats.items():
        print(f"  {dataset}: {stats[0]['name']} (gap = {stats[0]['mean']:.4f})")
    
    print("\nFiles generated:")
    print("  - acquisition_comparison.png")
    print("  - kappa_sensitivity.png")


if __name__ == "__main__":
    main()
