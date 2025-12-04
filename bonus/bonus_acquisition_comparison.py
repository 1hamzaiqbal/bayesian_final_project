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
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern


def branin(x1, x2, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    """Branin function (minimization)."""
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    return term1 + term2 + s


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
    """Compute gap metric for minimization."""
    denominator = f_best_initial - f_optimum
    if abs(denominator) < 1e-10:
        return 1.0 if f_best_found <= f_optimum else 0.0
    gap = (f_best_initial - f_best_found) / denominator
    return np.clip(gap, 0, 1)


def bayesian_optimization_with_acq(X_pool, y_pool, acquisition_fn, acq_params,
                                    n_initial=5, n_iterations=30, 
                                    use_log_transform=False):
    """
    Run Bayesian optimization with a specified acquisition function.
    """
    n_pool = len(X_pool)
    available = np.ones(n_pool, dtype=bool)
    
    selected_indices = []
    y_values = []
    best_so_far = []
    
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
        
        gp = fit_gp(X_train, y_train_transformed)
        
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
        'best_so_far': best_so_far
    }


def run_acquisition_comparison(n_runs=20, save_dir=None):
    """
    Compare EI, PI, and LCB acquisition functions.
    """
    print("=" * 70)
    print("ACQUISITION FUNCTION COMPARISON")
    print("=" * 70)
    
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
    
    datasets = {
        'Branin': (X_branin, y_branin, f_opt_branin, True),
        'LDA': (lda_data[:, :3], lda_data[:, 3], lda_data[:, 3].min(), True),
        'SVM': (svm_data[:, :3], svm_data[:, 3], svm_data[:, 3].min(), True)
    }
    
    # Acquisition functions to compare
    acquisition_configs = {
        'EI': {'xi': 0.01},
        'PI': {'xi': 0.01},
        'LCB (κ=1)': {'kappa': 1.0},
        'LCB (κ=2)': {'kappa': 2.0},
        'LCB (κ=3)': {'kappa': 3.0},
    }
    
    all_results = {}
    
    for name, (X_pool, y_pool, f_opt, use_log) in datasets.items():
        print(f"\n{'='*50}")
        print(f"  {name} DATASET")
        print(f"{'='*50}")
        
        dataset_results = {}
        
        for acq_name, acq_params in acquisition_configs.items():
            print(f"\n  Testing {acq_name}...")
            
            # Determine base acquisition function
            if acq_name.startswith('LCB'):
                acq_fn = 'LCB'
            else:
                acq_fn = acq_name
            
            runs = []
            for run in range(n_runs):
                np.random.seed(run * 42)
                
                history = bayesian_optimization_with_acq(
                    X_pool, y_pool, acq_fn, acq_params,
                    n_initial=5, n_iterations=30,
                    use_log_transform=use_log
                )
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
    
    return all_results


def plot_comparison_results(all_results, save_dir):
    """Create comparison plots for acquisition functions."""
    print("\n" + "=" * 70)
    print("CREATING COMPARISON PLOTS")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = {
        'EI': '#1f77b4',
        'PI': '#ff7f0e', 
        'LCB (κ=1)': '#2ca02c',
        'LCB (κ=2)': '#d62728',
        'LCB (κ=3)': '#9467bd'
    }
    
    for idx, (dataset_name, data) in enumerate(all_results.items()):
        ax = axes[idx]
        f_opt = data['f_opt']
        
        for acq_name, runs in data['results'].items():
            n_obs = len(runs[0]['best_so_far'])
            gaps = np.zeros((len(runs), n_obs))
            
            for run_idx, run in enumerate(runs):
                f_best_initial = run['best_so_far'][4]
                for i, best in enumerate(run['best_so_far']):
                    gaps[run_idx, i] = compute_gap(best, f_best_initial, f_opt)
            
            x = np.arange(1, n_obs + 1)
            mean_gap = gaps.mean(axis=0)
            std_gap = gaps.std(axis=0)
            
            ax.plot(x, mean_gap, label=acq_name, color=colors[acq_name], linewidth=2)
            ax.fill_between(x, mean_gap - std_gap, mean_gap + std_gap, 
                           alpha=0.15, color=colors[acq_name])
        
        ax.set_xlabel('Number of Observations')
        ax.set_ylabel('Gap (higher is better)')
        ax.set_title(f'{dataset_name} Dataset')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim([1, 35])
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
    
    summary_stats = {}
    
    for dataset_name, data in all_results.items():
        print(f"\n{'='*50}")
        print(f"  {dataset_name} DATASET")
        print(f"{'='*50}")
        
        f_opt = data['f_opt']
        stats_list = []
        
        for acq_name, runs in data['results'].items():
            final_gaps = []
            for run in runs:
                f_best_initial = run['best_so_far'][4]
                f_best_found = run['best_so_far'][-1]
                gap = compute_gap(f_best_found, f_best_initial, f_opt)
                final_gaps.append(gap)
            
            final_gaps = np.array(final_gaps)
            stats_list.append({
                'name': acq_name,
                'mean': final_gaps.mean(),
                'std': final_gaps.std(),
                'gaps': final_gaps
            })
        
        # Sort by mean gap (descending)
        stats_list.sort(key=lambda x: x['mean'], reverse=True)
        
        print(f"\n  Rankings (by mean gap):")
        print(f"  {'Rank':<5} {'Acquisition':<15} {'Mean Gap':<12} {'Std':<10}")
        print("  " + "-" * 45)
        
        for rank, stat in enumerate(stats_list):
            print(f"  {rank+1:<5} {stat['name']:<15} {stat['mean']:<12.4f} {stat['std']:<10.4f}")
        
        # Paired t-tests: compare best against others
        best = stats_list[0]
        print(f"\n  Paired t-tests ({best['name']} vs others):")
        
        for stat in stats_list[1:]:
            t_stat, p_value = stats.ttest_rel(best['gaps'], stat['gaps'])
            sig = "*" if p_value < 0.05 else ""
            print(f"    vs {stat['name']:<12}: t={t_stat:7.3f}, p={p_value:.4f} {sig}")
        
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
    
    kappa_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    n_runs = 20
    
    results_by_kappa = {}
    
    for kappa in kappa_values:
        print(f"  Testing κ = {kappa}...")
        
        gaps = []
        for run in range(n_runs):
            np.random.seed(run * 42)
            
            history = bayesian_optimization_with_acq(
                X_pool, y_pool, 'LCB', {'kappa': kappa},
                n_initial=5, n_iterations=30,
                use_log_transform=True
            )
            
            f_best_initial = history['best_so_far'][4]
            f_best_found = history['best_so_far'][-1]
            gap = compute_gap(f_best_found, f_best_initial, f_opt)
            gaps.append(gap)
        
        results_by_kappa[kappa] = np.array(gaps)
    
    # Plot
    kappas = list(results_by_kappa.keys())
    means = [results_by_kappa[k].mean() for k in kappas]
    stds = [results_by_kappa[k].std() for k in kappas]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(kappas, means, yerr=stds, marker='o', capsize=5, 
                linewidth=2, markersize=8, color='#d62728')
    
    ax.set_xlabel('κ (exploration-exploitation trade-off)', fontsize=11)
    ax.set_ylabel('Mean Gap (±std)', fontsize=11)
    ax.set_title('LCB Sensitivity to κ on Branin Function', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])
    
    # Mark optimal κ
    best_idx = np.argmax(means)
    ax.annotate(f'Best: κ={kappas[best_idx]}', 
                xy=(kappas[best_idx], means[best_idx]),
                xytext=(kappas[best_idx] + 0.5, means[best_idx] - 0.1),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kappa_sensitivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: kappa_sensitivity.png")
    
    print(f"\n  Optimal κ = {kappas[best_idx]} with mean gap = {means[best_idx]:.4f}")
    
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
