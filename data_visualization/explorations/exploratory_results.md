# Data Visualization Explorations

This folder contains extra experiments that *question and extend* the assumptions made in `data_visualization/report.md`, focused on making the objectives look more “stationary” (i.e., easier to model with a stationary GP kernel).

---

## Branin: output transforms and “stationarity”

Baseline report claim: `log(f+1)` improves stationarity by compressing the dynamic range and reducing gradient variation.

We tested multiple **monotone variance‑compressing transforms** and measured a proxy for stationarity: **how uniformly the gradient magnitude varies across the domain** (robust ratios of percentiles; lower is better).

Run:

`python data_visualization/explorations/branin_stationarity_transform_sweep.py`

Output:

- `exp_branin_transform_stationarity_grid.png`

Results (gradient‑magnitude variation):

| Transform | p95/p5 of \|∇F\| ↓ | Notes |
|---|---:|---|
| identity | 15.00 | Highly non-uniform local curvature |
| log(f+1) | 9.35 | Improves, but still quite non-uniform |
| **sqrt(f)** | **5.16** | Best of the three by this metric |

Takeaway:

- `sqrt(f)` makes the **local gradient variation substantially more uniform** than `log(f+1)` on this domain, suggesting it may be a stronger “stationarizing” output transform for Branin.
- `log(f+1)` still compresses the raw value range more aggressively; the “best” choice depends on whether you care more about amplitude compression or curvature uniformity.

---

## LDA/SVM: output transforms (distribution shape)

Baseline report: `log(y)` reduces right‑skewness.

We compared simple log transforms to a learned **Box‑Cox power transform**, plotting **z‑scored** transformed outputs (so shapes are comparable).

Run:

`python data_visualization/explorations/benchmark_output_transform_sweep.py`

Output:

- `exp_benchmark_output_transform_zscore_kde.png`

Key result:

- Box‑Cox makes both LDA and SVM distributions substantially more symmetric than log (skew drops to ~0.3).

Important caveat:

- GP regression does **not** require the marginal distribution of y across different x’s to be Gaussian; these transforms are mainly about **reducing extreme leverage** and making modeling easier.

---

## LDA/SVM: input scaling (hyperparameter space)

This is the big one: the hyperparameter grids span **orders of magnitude** in some dimensions (e.g., SVM has a dimension from 0.1 to 1e6). Modeling in raw coordinates can look highly non‑stationary purely due to scaling.

We computed a stationarity proxy on the full 3D grids: the distribution of `|∇f|` under **raw inputs** vs **log10‑scaled inputs** on the high‑dynamic‑range dimensions.

Run:

`python data_visualization/explorations/benchmark_input_scaling_stationarity.py`

Output:

- `exp_benchmark_input_scaling_gradmag_kde.png`

Results (p95/p5 of |∇f|, lower is better):

| Dataset | Raw inputs | Log-scaled inputs | Improvement |
|---|---:|---:|---:|
| LDA | 245.75 | 90.10 (log10 dims 2&3) | 2.7× |
| SVM | 2112.50 | 22.98 (log10 dims 1&3) | 92× |

Takeaway:

- For LDA/SVM, **log-scaling the hyperparameter inputs** is arguably the most impactful “stationarizing” transform to consider for GP modeling.
