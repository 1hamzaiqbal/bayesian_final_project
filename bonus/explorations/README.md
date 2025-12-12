# Bonus explorations

This folder contains “unshackled” follow‑ups to the bonus acquisition comparison, focused on challenging modeling assumptions (kernel, transformation, noise handling) and checking parameter sensitivity.

## What changed vs the original bonus setup

- The original bonus comparison used **RBF + log(y+1)** everywhere with a fixed `alpha = 0.001²` under `normalize_y=True` (which does *not* correspond to σ=0.001 in original units).
- Here we explicitly test and then adopt:
  - **Correct noise scaling** (so σ=0.001 is in original units under `normalize_y=True`)
  - **Branin:** original scale + **SE + Periodic(x1)** (matches the `cos(x1)` structure)
  - **LDA/SVM:** **Matern 3/2** + log(y+1)

## Scripts

- `acquisition_modeling_sweep.py`: compares acquisition performance under multiple modeling configs; outputs:
  - `modeling_sweep_final_gap.png`
  - `modeling_sweep_auc.png`
- `acquisition_param_sensitivity.py`: parameter sweeps under the improved model; outputs:
  - `svm_kappa_sensitivity_improved.png`
  - `lda_pi_xi_sensitivity_improved.png`

## Key takeaways

- Kernel/transform/noise choices can change the “winning acquisition” more than the acquisition function itself.
- Under the improved modeling setup:
  - Branin results saturate and EI/PI/LCB become nearly indistinguishable by final gap.
  - SVM strongly prefers **LCB with κ≈1**.
  - LDA prefers **PI with ξ≈0.01**.

