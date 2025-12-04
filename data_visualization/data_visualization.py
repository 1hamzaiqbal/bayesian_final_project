"""
Data Visualization for Bayesian Optimization Project

This script addresses all bullet points from the data visualization section:
1. Branin function heatmap (1000x1000 grid)
2. Stationarity analysis
3. Transformation for stationarity
4. Kernel density estimates for LDA and SVM benchmarks
5. Transformation for better-behaved distributions

Author: Generated for Bayesian Optimization final project
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os


def branin(x1, x2, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    """
    Branin (Branin-Hoo) function.
    
    Parameters:
    -----------
    x1, x2 : array-like
        Input coordinates. Domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    a, b, c, r, s, t : float
        Function parameters with standard default values
        
    Returns:
    --------
    y : array-like
        Function values
        
    Notes:
    ------
    The function has 3 global minima with value f* ≈ 0.397887 at:
    - (-π, 12.275)
    - (π, 2.275)
    - (9.42478, 2.475)
    """
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    return term1 + term2 + s


def create_branin_heatmap(save_path=None):
    """
    Create a 1000x1000 heatmap of the Branin function.
    
    Bullet point 1: Make a heatmap of the value of the Branin function 
    over the domain X = [−5, 10] × [0, 15] using a dense grid of values, 
    with 1000 values per dimension, forming a 1000 × 1000 image.
    """
    print("=" * 70)
    print("BULLET POINT 1: Branin Function Heatmap")
    print("=" * 70)
    
    # Create 1000x1000 grid
    x1 = np.linspace(-5, 10, 1000)
    x2 = np.linspace(0, 15, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Evaluate Branin function
    Z = branin(X1, X2)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(Z, extent=[-5, 10, 0, 15], origin='lower', 
                   aspect='auto', cmap='viridis')
    
    # Mark the three global minima
    minima = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
    for x, y in minima:
        ax.plot(x, y, 'r*', markersize=15, markeredgecolor='white', 
                markeredgewidth=1.5, label='Global minimum' if x == -np.pi else '')
    
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Branin Function Heatmap\n$f(x_1, x_2) = a(x_2 - bx_1^2 + cx_1 - r)^2 + s(1-t)\\cos(x_1) + s$', 
                 fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('$f(x_1, x_2)$', fontsize=11)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    
    print(f"\nGrid: {X1.shape[0]} × {X1.shape[1]} = 1,000,000 points")
    print(f"Domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]")
    print(f"Minimum value: {Z.min():.6f}")
    print(f"Maximum value: {Z.max():.6f}")
    
    return X1, X2, Z


def analyze_stationarity():
    """
    Bullet point 2: Describe the behavior of the function. Does it appear 
    stationary? (That is, does the behavior of the function appear to be 
    relatively constant throughout the domain?)
    """
    print("\n" + "=" * 70)
    print("BULLET POINT 2: Stationarity Analysis")
    print("=" * 70)
    
    analysis = """
The Branin function is NON-STATIONARY. This is evident from several observations:

1. VARYING MAGNITUDE: The function values range from approximately 0.4 (at the 
   three global minima) to over 300 at the domain corners. This dramatic range
   indicates non-constant behavior.

2. ASYMMETRIC STRUCTURE: The function has three distinct global minima located at:
   - (-π, 12.275)
   - (π, 2.275)  
   - (9.42478, 2.475)
   These create valleys of varying depths and widths across the domain.

3. QUADRATIC COMPONENT: The term a(x2 - bx1² + cx1 - r)² creates a parabolic 
   valley structure that changes curvature across the domain.

4. PERIODIC MODULATION: The cosine term s(1-t)cos(x1) adds periodic oscillation
   in the x1 direction only, creating wave-like patterns that interact with 
   the quadratic structure.

5. EDGE EFFECTS: Function values are much higher near the domain boundaries,
   especially at the corners, compared to the interior regions near the minima.

IMPLICATIONS FOR BAYESIAN OPTIMIZATION:
- A stationary GP prior (constant mean, stationary kernel) may struggle to 
  capture the varying behavior across the domain.
- The optimizer may need more samples in high-variance regions.
- Adaptive lengthscale or non-stationary kernels may improve performance.
"""
    print(analysis)
    return analysis


def create_transformed_heatmap(save_path=None):
    """
    Bullet point 3: Can you find a transformation of the data that makes 
    it more stationary?
    """
    print("\n" + "=" * 70)
    print("BULLET POINT 3: Transformation for Stationarity")
    print("=" * 70)
    
    # Create grid
    x1 = np.linspace(-5, 10, 1000)
    x2 = np.linspace(0, 15, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Z = branin(X1, X2)
    
    # Apply log transformation (adding 1 to handle near-zero values)
    Z_log = np.log(Z + 1)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original
    im1 = axes[0].imshow(Z, extent=[-5, 10, 0, 15], origin='lower', 
                         aspect='auto', cmap='viridis')
    axes[0].set_xlabel('$x_1$', fontsize=11)
    axes[0].set_ylabel('$x_2$', fontsize=11)
    axes[0].set_title('Original Branin Function', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('$f(x_1, x_2)$', fontsize=10)
    
    # Log-transformed
    im2 = axes[1].imshow(Z_log, extent=[-5, 10, 0, 15], origin='lower', 
                         aspect='auto', cmap='viridis')
    axes[1].set_xlabel('$x_1$', fontsize=11)
    axes[1].set_ylabel('$x_2$', fontsize=11)
    axes[1].set_title('Log-Transformed: $\\log(f + 1)$', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('$\\log(f + 1)$', fontsize=10)
    
    plt.suptitle('Transformation to Improve Stationarity', fontsize=13, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    
    explanation = """
LOG TRANSFORMATION: log(f + 1)

The log transformation helps make the function more stationary by:

1. COMPRESSING DYNAMIC RANGE: The original range [0.4, 300+] becomes [0.34, 5.7],
   reducing the ratio of max to min from ~750x to ~17x.

2. REDUCING EDGE EXPLOSION: The extreme values at domain boundaries are 
   compressed, making the function appear more uniform.

3. PRESERVING STRUCTURE: The locations of minima and general shape are 
   preserved while reducing the magnitude of variations.

4. APPROXIMATE VARIANCE STABILIZATION: The log transform is a variance-
   stabilizing transformation for positive-valued data.

Original statistics:
  - Range: {:.2f} to {:.2f}
  - Std dev: {:.2f}

Log-transformed statistics:
  - Range: {:.2f} to {:.2f}  
  - Std dev: {:.2f}
""".format(Z.min(), Z.max(), Z.std(), Z_log.min(), Z_log.max(), Z_log.std())
    
    print(explanation)
    
    return Z, Z_log


def load_benchmark_data():
    """Load LDA and SVM benchmark datasets."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Load CSV files (no headers)
    lda_path = os.path.join(parent_dir, 'lda.csv')
    svm_path = os.path.join(parent_dir, 'svm.csv')
    
    lda_data = pd.read_csv(lda_path, header=None)
    svm_data = pd.read_csv(svm_path, header=None)
    
    # Column 4 (index 3) is the objective value to minimize
    lda_values = lda_data.iloc[:, 3].values
    svm_values = svm_data.iloc[:, 3].values
    
    return lda_values, svm_values


def create_kde_plots(save_path=None):
    """
    Bullet point 4: Make a kernel density estimate of the distribution of 
    the values for the lda and svm benchmarks. Interpret the distributions.
    """
    print("\n" + "=" * 70)
    print("BULLET POINT 4: Kernel Density Estimates")
    print("=" * 70)
    
    lda_values, svm_values = load_benchmark_data()
    
    # Create KDE plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LDA KDE
    x_lda = np.linspace(lda_values.min() * 0.9, lda_values.max() * 1.1, 1000)
    kde_lda = stats.gaussian_kde(lda_values)
    axes[0].fill_between(x_lda, kde_lda(x_lda), alpha=0.5, color='steelblue')
    axes[0].plot(x_lda, kde_lda(x_lda), color='steelblue', linewidth=2)
    axes[0].axvline(lda_values.min(), color='red', linestyle='--', 
                    label=f'Min: {lda_values.min():.2f}', linewidth=1.5)
    axes[0].set_xlabel('Performance Value (to minimize)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('LDA Benchmark: Kernel Density Estimate', fontsize=12)
    axes[0].legend()
    
    # SVM KDE
    x_svm = np.linspace(svm_values.min() * 0.9, svm_values.max() * 1.1, 1000)
    kde_svm = stats.gaussian_kde(svm_values)
    axes[1].fill_between(x_svm, kde_svm(x_svm), alpha=0.5, color='darkorange')
    axes[1].plot(x_svm, kde_svm(x_svm), color='darkorange', linewidth=2)
    axes[1].axvline(svm_values.min(), color='red', linestyle='--', 
                    label=f'Min: {svm_values.min():.4f}', linewidth=1.5)
    axes[1].set_xlabel('Performance Value (to minimize)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('SVM Benchmark: Kernel Density Estimate', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    
    interpretation = """
INTERPRETATION OF DISTRIBUTIONS:

LDA BENCHMARK:
- Sample size: {} hyperparameter configurations
- Range: [{:.2f}, {:.2f}]
- Mean: {:.2f}, Std: {:.2f}
- Shape: RIGHT-SKEWED distribution with a long tail toward higher values
- Interpretation: Most hyperparameter configurations yield moderate performance,
  but some configurations produce very poor results. The optimal configurations
  are relatively rare (left tail). This suggests the hyperparameter landscape
  has a few good regions and many suboptimal regions.

SVM BENCHMARK:
- Sample size: {} hyperparameter configurations  
- Range: [{:.4f}, {:.4f}]
- Mean: {:.4f}, Std: {:.4f}
- Shape: RIGHT-SKEWED with a pronounced mode around 0.27-0.35
- Interpretation: The SVM error rate clusters around 0.27-0.35, suggesting
  many configurations achieve similar moderate performance. There are outliers
  at 0.5 (random chance for classification), indicating poor configurations.
  The minimum around 0.25 represents the best achievable performance.

BOTH DISTRIBUTIONS ARE RIGHT-SKEWED:
- This is common in hyperparameter optimization: there are many ways to 
  configure a model poorly but fewer ways to configure it optimally.
- The long right tails suggest transformations may help for GP modeling.
""".format(len(lda_values), lda_values.min(), lda_values.max(), 
           lda_values.mean(), lda_values.std(),
           len(svm_values), svm_values.min(), svm_values.max(),
           svm_values.mean(), svm_values.std())
    
    print(interpretation)
    
    return lda_values, svm_values


def create_transformed_kde_plots(save_path=None):
    """
    Bullet point 5: Again, can you find a transformation that makes the 
    performance better behaved?
    """
    print("\n" + "=" * 70)
    print("BULLET POINT 5: Transformation for Better-Behaved Distributions")
    print("=" * 70)
    
    lda_values, svm_values = load_benchmark_data()
    
    # Apply log transformation
    lda_log = np.log(lda_values)
    svm_log = np.log(svm_values)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # LDA Original
    x_lda = np.linspace(lda_values.min() * 0.9, lda_values.max() * 1.1, 1000)
    kde_lda = stats.gaussian_kde(lda_values)
    axes[0, 0].fill_between(x_lda, kde_lda(x_lda), alpha=0.5, color='steelblue')
    axes[0, 0].plot(x_lda, kde_lda(x_lda), color='steelblue', linewidth=2)
    axes[0, 0].set_xlabel('Performance Value', fontsize=10)
    axes[0, 0].set_ylabel('Density', fontsize=10)
    axes[0, 0].set_title('LDA: Original Distribution', fontsize=11)
    
    # LDA Log-transformed
    x_lda_log = np.linspace(lda_log.min() * 0.95, lda_log.max() * 1.05, 1000)
    kde_lda_log = stats.gaussian_kde(lda_log)
    axes[0, 1].fill_between(x_lda_log, kde_lda_log(x_lda_log), alpha=0.5, color='steelblue')
    axes[0, 1].plot(x_lda_log, kde_lda_log(x_lda_log), color='steelblue', linewidth=2)
    axes[0, 1].set_xlabel('log(Performance Value)', fontsize=10)
    axes[0, 1].set_ylabel('Density', fontsize=10)
    axes[0, 1].set_title('LDA: Log-Transformed Distribution', fontsize=11)
    
    # SVM Original
    x_svm = np.linspace(svm_values.min() * 0.9, svm_values.max() * 1.1, 1000)
    kde_svm = stats.gaussian_kde(svm_values)
    axes[1, 0].fill_between(x_svm, kde_svm(x_svm), alpha=0.5, color='darkorange')
    axes[1, 0].plot(x_svm, kde_svm(x_svm), color='darkorange', linewidth=2)
    axes[1, 0].set_xlabel('Performance Value', fontsize=10)
    axes[1, 0].set_ylabel('Density', fontsize=10)
    axes[1, 0].set_title('SVM: Original Distribution', fontsize=11)
    
    # SVM Log-transformed
    x_svm_log = np.linspace(svm_log.min() * 0.95, svm_log.max() * 1.05, 1000)
    kde_svm_log = stats.gaussian_kde(svm_log)
    axes[1, 1].fill_between(x_svm_log, kde_svm_log(x_svm_log), alpha=0.5, color='darkorange')
    axes[1, 1].plot(x_svm_log, kde_svm_log(x_svm_log), color='darkorange', linewidth=2)
    axes[1, 1].set_xlabel('log(Performance Value)', fontsize=10)
    axes[1, 1].set_ylabel('Density', fontsize=10)
    axes[1, 1].set_title('SVM: Log-Transformed Distribution', fontsize=11)
    
    plt.suptitle('Comparison: Original vs Log-Transformed Distributions', fontsize=13, y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    
    # Calculate skewness
    lda_skew_orig = stats.skew(lda_values)
    lda_skew_log = stats.skew(lda_log)
    svm_skew_orig = stats.skew(svm_values)
    svm_skew_log = stats.skew(svm_log)
    
    explanation = """
LOG TRANSFORMATION ANALYSIS:

The log transformation log(y) helps make the distributions more symmetric
and closer to Gaussian, which is beneficial for GP modeling.

SKEWNESS COMPARISON (0 = symmetric, >0 = right-skewed, <0 = left-skewed):

LDA Benchmark:
  - Original skewness: {:.3f}
  - Log-transformed skewness: {:.3f}
  - Improvement: {:.1f}% reduction in skewness magnitude

SVM Benchmark:
  - Original skewness: {:.3f}
  - Log-transformed skewness: {:.3f}
  - Improvement: {:.1f}% reduction in skewness magnitude

BENEFITS OF LOG TRANSFORMATION:

1. SYMMETRY: Both distributions become more symmetric after log transformation,
   with skewness closer to zero.

2. VARIANCE STABILIZATION: The variance becomes more uniform across the range
   of values, which helps satisfy GP assumptions.

3. OUTLIER COMPRESSION: Extreme high values are compressed, reducing their
   influence on the GP fit.

4. NORMAL APPROXIMATION: The log-transformed distributions are closer to
   Gaussian, making the GP likelihood assumption more appropriate.

RECOMMENDATION: Use log-transformed objective values when fitting GPs for
Bayesian optimization on these benchmarks.
""".format(lda_skew_orig, lda_skew_log, 
           100 * (1 - abs(lda_skew_log) / abs(lda_skew_orig)),
           svm_skew_orig, svm_skew_log,
           100 * (1 - abs(svm_skew_log) / abs(svm_skew_orig)))
    
    print(explanation)
    
    return lda_log, svm_log


def main():
    """Run all data visualization tasks."""
    print("\n" + "=" * 70)
    print("DATA VISUALIZATION FOR BAYESIAN OPTIMIZATION PROJECT")
    print("=" * 70)
    
    # Get output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Bullet Point 1: Branin Heatmap
    create_branin_heatmap(
        save_path=os.path.join(output_dir, 'branin_heatmap.png')
    )
    
    # Bullet Point 2: Stationarity Analysis
    analyze_stationarity()
    
    # Bullet Point 3: Transformation for Stationarity
    create_transformed_heatmap(
        save_path=os.path.join(output_dir, 'branin_transformed_heatmap.png')
    )
    
    # Bullet Point 4: KDE for LDA and SVM
    create_kde_plots(
        save_path=os.path.join(output_dir, 'kde_lda_svm.png')
    )
    
    # Bullet Point 5: Transformation for Better Distributions
    create_transformed_kde_plots(
        save_path=os.path.join(output_dir, 'kde_lda_svm_transformed.png')
    )
    
    print("\n" + "=" * 70)
    print("SUMMARY: All visualizations completed!")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - branin_heatmap.png")
    print("  - branin_transformed_heatmap.png")
    print("  - kde_lda_svm.png")
    print("  - kde_lda_svm_transformed.png")
    print("\nAll 5 bullet points from the data visualization section addressed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
