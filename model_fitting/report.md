# Model Fitting

This section evaluates whether we can build useful Gaussian process models of our objective functions before using them for optimization.

---

## Bullet Point Reference

This report addresses each bullet point from the [instructions](instructions.md):

| Bullet | Instruction Summary | Report Section |
|--------|---------------------|----------------|
| 1 | Generate 32 Sobol training points | [Section 1](#1-training-data-generation) |
| 2 | Fit GP with constant mean + SE kernel, noise=0.001 | [Section 2](#2-gp-model-with-squared-exponential-kernel) |
| 3 | Report learned hyperparameters | [Section 2](#learned-hyperparameters) |
| 4 | Heatmap of GP posterior mean | [Section 3](#3-posterior-mean-heatmap) |
| 5 | Heatmap of GP posterior std | [Section 4](#4-posterior-standard-deviation-heatmap) |
| 6 | Z-score KDE for calibration | [Section 5](#5-z-score-calibration-analysis) |
| 7 | Repeat with log transformation | [Section 6](#6-log-transformed-branin-analysis) |
| 8 | Compute BIC score | [Section 7](#7-bic-model-selection) |
| 9 | Search over models to find best BIC | [Section 8](#8-model-search-results) |
| 10 | Model search for SVM and LDA | [Section 9](#9-real-benchmark-model-search) |

---

## 1. Training Data Generation

> **Bullet 1:** *"Select a set of 32 training points for the Branin function in the domain X = [−5, 10] × [0, 15] using a Sobol sequence."*

We generated 32 training points using a **Sobol sequence** over $\mathcal{X} = [-5, 10] \times [0, 15]$.

```
Training set statistics:
  - Number of points: 32
  - Domain: x₁ ∈ [-5, 10], x₂ ∈ [0, 15]
  - y range: [2.18, 242.12]
  - y mean: 55.26, y std: 55.03
```

---

## 2. GP Model with Squared Exponential Kernel

> **Bullet 2:** *"Fit a Gaussian process model to the data using a constant mean and a squared exponential covariance. Fix the standard deviation of the noise to 0.001. Maximize the marginal likelihood."*

We fit a Gaussian process with:
- **Mean function:** Constant (learned via `normalize_y=True`)
- **Covariance function:** ARD Squared Exponential (RBF with per-dimension lengthscales)
- **Noise:** Fixed at σ = 0.001 (alpha = σ² = 10⁻⁶)

### Learned Hyperparameters

> **Bullet 3:** *"What values did you learn for the hyperparameters? Do they agree with your expectations given your visualization?"*

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| **Constant mean** | 55.26 | ≈ empirical mean of y ✓ |
| Output scale | 15.2² = 231 | Overall function variance |
| Length scale (x₁) | 4.75 | Correlation decays over ~5 units |
| Length scale (x₂) | 39.5 | Very long-range correlation ⚠ |
| Log marginal likelihood | 24.71 | Model fit quality |

> **⚠ NOTE on ℓ₂:** The length scale ℓ₂ ≈ 39.5 is **> 2× the domain width** [0, 15]. This suggests the model thinks f varies slowly in x₂, possibly indicating local optimum in MLL optimization or kernel misspecification. The Branin function has a quadratic dependence on x₂, so this may oversmooth.

**Do they agree with expectations?** Partially:
- x₁ has moderate length scale (~5), reflecting the cosine oscillation with wavelength 2π ≈ 6.3 ✓
- x₂ length scale is suspiciously large—flagged as potential issue ⚠
- The large output scale accounts for the high function range (0.4 to 300+) ✓

---

## 3. Posterior Mean Heatmap

> **Bullet 4:** *"Make a heatmap of the Gaussian process posterior mean. Compare the predicted values with the true values. Do you see systematic errors?"*

![GP Posterior Heatmaps - Original](original_posterior_heatmaps.png)

**Figure 1:** Left: True Branin. Middle: GP posterior mean. Right: GP posterior std.

### Residual Analysis

To quantify systematic errors, we visualize the residuals $\mu(x) - f(x)$:

![Residual Heatmap](original_residual_heatmap.png)

**Figure 1b:** Residual heatmap showing $\mu(x) - f(x)$. Red = overprediction, Blue = underprediction.

**Do you see systematic errors?**
The residual heatmap shows some systematic patterns, particularly nonzero errors in corner regions far from training data. However, the RMSE is reasonable and the GP captures the overall structure well.

---

## 4. Posterior Standard Deviation Heatmap

> **Bullet 5:** *"Make a heatmap of the Gaussian process posterior standard deviation."*

| Metric | Value |
|--------|-------|
| Min σ (at training points) | 0.042 |
| Max σ (at training points) | 0.055 |
| Mean σ (at training points) | 0.051 |

### Important Clarification on σ(x)

The σ values shown are **predictive standard deviation** (includes noise variance). With `normalize_y=True`, sklearn internally rescales outputs to mean 0, std 1, then rescales predictions back. This means:

- Effective noise in original units ≈ 0.001 × 55.0 ≈ **0.055**
- The σ ≈ 0.04-0.055 at training points is consistent with this

**Answers:**
- ✓ **Does σ drop to near zero at data points?** Yes, σ ≈ 0.04-0.055 (consistent with rescaled noise)
- ✓ **Does the scale make sense?** Yes, σ ranges from ~0.05 (at data) to higher values far from data

---

## 5. Z-Score Calibration Analysis

> **Bullet 6:** *"Make a kernel density estimate of the z-scores. Is the GP model well calibrated?"*

![Z-Score Distribution - Original](zscore_original.png)

**Figure 2:** KDE of z-scores with coverage metrics.

### Coverage Metrics

| Metric | Value | Target | Assessment |
|--------|-------|--------|------------|
| Z-score mean | -0.035 | 0 | ✓ No systematic bias |
| Z-score std | 1.228 | 1 | Slightly overconfident |
| Coverage \|z\|≤1 | 52.0% | 68.3% | ⚠ Undercovers |
| Coverage \|z\|≤2 | 91.9% | 95.4% | Close to target |

### Is the GP model well calibrated?

**Partially.** The mean is near zero (no bias), but the std = 1.228 > 1 indicates the model is **slightly overconfident**—the GP's uncertainty estimates are narrower than they should be. The 52% coverage at |z|≤1 (vs target 68.3%) confirms this.

Note: The KDE shape shows some deviation from Gaussian (slight bimodality), suggesting residual structure not captured by the stationary kernel.

---

## 6. Log-Transformed Branin Analysis

> **Bullet 7:** *"Repeat the above using a log transformation. Does the marginal likelihood improve? Does the model appear better calibrated?"*

![GP Posterior Heatmaps - Log Transformed](log_transformed_posterior_heatmaps.png)

### Marginal Likelihood Comparison

| Model | Log Marginal Likelihood |
|-------|-------------------------|
| Original | 24.71 |
| Log-transformed | -19.48 |

**Note:** These values are not directly comparable due to the transformation. To properly compare, one would need to include the Jacobian term: $\log p(y|x) = \log p(y'|x) - \log(y+1)$.

### Calibration (Log-Transformed)

![Z-Score Distribution - Log Transformed](zscore_log_transformed.png)

| Metric | Original | Log-Transformed |
|--------|----------|-----------------|
| Mean | -0.035 | 0.027 |
| Std | 1.228 | **2.234** |
| Coverage \|z\|≤1 | 52.0% | 45.2% |
| Coverage \|z\|≤2 | 91.9% | 70.2% |

**Is the log-transformed model better calibrated?**

**No, it is worse.** The std = 2.234 indicates the model is **significantly overconfident** (uncertainty too narrow). The 45% coverage at |z|≤1 (vs target 68.3%) confirms this.

---

## 7. BIC Model Selection

> **Bullet 8:** *"Compute the BIC score for the data and model from the last part."*

$$\text{BIC} = k \log n - 2 \log \hat{\mathcal{L}}$$

### BIC Calculation

```
k = 3 (kernel parameters: output_scale, ℓ₁, ℓ₂)
n = 32
BIC = 3 × log(32) - 2 × (-19.48) = 49.36
```

**Note:** The constant mean is handled by `normalize_y=True` and is not counted in k here. If explicitly counted, k = 4.

---

## 8. Model Search Results

> **Bullet 9:** *"Search over models to find the best BIC."*

| Rank | Kernel | BIC | Log-Likelihood | k |
|------|--------|-----|----------------|---|
| **1** | **SE (RBF)** | **49.36** | -19.48 | 3 |
| 2 | Matern 5/2 | 51.91 | -20.76 | 3 |
| 3 | Matern 3/2 | 56.60 | -23.10 | 3 |
| 4 | SE (isotropic) | 66.16 | -29.61 | 2 |
| 5 | RationalQuadratic | 68.38 | -28.99 | 3 |
| 6 | SE + Matern 5/2 | 71.62 | -28.88 | 4 |

**Best Model:** SE (RBF), BIC = 49.36

**Note:** We did not explore periodic kernels. Given Branin contains a cos(x₁) term, adding a periodic component in x₁ could improve fit.

---

## 9. Real Benchmark Model Search

> **Bullet 10:** *"Model search for SVM and LDA datasets."*

### LDA Benchmark
| Rank | Kernel | BIC |
|------|--------|-----|
| **1** | **Matern 3/2** | **63.31** |
| 2 | Matern 5/2 | 65.60 |
| 3 | SE (RBF) | 70.45 |

### SVM Benchmark
| Rank | Kernel | BIC |
|------|--------|-----|
| **1** | **Matern 3/2** | **64.00** |
| 2 | Matern 5/2 | 65.92 |
| 3 | SE (RBF) | 70.49 |

**Interpretation:** Real hyperparameter surfaces prefer **Matérn 3/2** (rougher) over SE (smooth).

---

## Summary

| Bullet | Question | Answer |
|--------|----------|--------|
| 3 | Hyperparameters agree with expectations? | Partially—ℓ₂ suspiciously large ⚠ |
| 4 | Systematic errors? | Some in corners (see residual heatmap) |
| 5 | σ drop to ~0 at data points? | Yes—consistent with rescaled noise |
| 6 | Z-scores ~ N(0,1)? | Partially—std 1.23, slight overconfidence |
| 7 | Log transform improves calibration? | **No**—std 2.23, significantly overconfident |
| 8 | BIC for log-transformed SE | 49.36 |
| 9 | Best Branin model | SE (RBF), BIC=49.36 |
| 10 | Best LDA/SVM models | Matern 3/2, BIC≈63-64 |

**Key findings:**
1. Original-scale GP is slightly overconfident (std 1.23)
2. Log-transformed GP is significantly overconfident (std 2.23)
3. ℓ₂ lengthscale may indicate misspecification
4. Real hyperparameter surfaces prefer rougher Matérn kernels
