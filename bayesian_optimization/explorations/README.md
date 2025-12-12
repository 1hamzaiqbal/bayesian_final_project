# Explorations: questioning BO assumptions on Branin

This folder contains extra experiments that revisit key Bayesian optimization choices from the main write‑up.

## Assumptions challenged

- **Noise level under `normalize_y=True`.** The original BO GP used `alpha = σ²` in *normalized* units, which inflated effective noise on Branin. We tested and then adopted corrected scaling so the intended σ=0.001 holds in original units.
- **Log transform for Branin.** Model fitting showed log(y+1) is worse calibrated for Branin. We re‑ran BO on the original scale.
- **Kernel misspecification.** Branin has an explicit periodic term in `x1`. We tried a compositional kernel **SE + Periodic(x1)**.

## Results

Run `python branin_bo_assumption_sweep.py` to reproduce.

Final gap at 35 evaluations (20 paired runs):

- Baseline log + SE: gap ≈ 0.990 ± 0.006  
- Log + SE with corrected noise: gap ≈ 0.995 ± 0.002  
- **Original scale + SE + Periodic(x1): gap ≈ 1.000 ± 0.000 (relative to pool optimum)**  

Learning curves comparing these variants are saved as `branin_bo_learning_comparison.png`.

**Takeaway:** Fixing noise scaling and using the structure‑matched periodic kernel yields the strongest Branin BO performance. This setup is now used in the main `bayesian_optimization.py` and report.

