# Explorations: Branin alternative GP models

Key assumptions questioned:

- **Noise level interpretation.** In `GaussianProcessRegressor(normalize_y=True)`, a fixed `alpha` is in normalized units. The baseline effectively used σ≈0.055 on Branin instead of σ=0.001. `model_fitting.py` now rescales `alpha` so σ=0.001 is respected in original units.
- **Kernel family.** Branin has a cosine term in `x1`, so purely stationary SE/Matérn kernels are misspecified. We explored periodic structure in `x1` using a dimension‑masked periodic kernel.

Results (from `branin_alternative_models.py`, evaluated on a 100×100 grid):

| Model | RMSE ↓ | NLPD ↓ | z‑std (target≈1) | cov\|z\|≤1 (target≈68%) |
|---|---:|---:|---:|---:|
| SE + Periodic(x1) | **0.05** | **‑2.87** | 1.62 | 55.6% |
| SE × Periodic(x1) | 0.13 | ‑1.80 | 1.60 | 40.1% |
| SE (ARD) baseline | 0.86 | ‑0.42 | 1.55 | 28.1% |

Takeaway:

- Adding **periodicity in `x1`** gives an order‑of‑magnitude improvement in mean prediction accuracy versus SE alone, and noticeably improves calibration, though uncertainty is still a bit narrow (z‑std ≈1.6).
- For later Bayesian optimization, this kernel is a much more faithful surrogate for Branin on the original scale.

Next directions if you want to push further:

- Try a **quasi‑periodic + polynomial trend** (e.g., periodic(x1) × SE + quadratic mean) to address remaining calibration error.
- For SVM/LDA benchmarks, consider **learning a noise term** (`WhiteKernel`) or using a larger fixed σ, since those datasets are not deterministic.

