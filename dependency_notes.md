The spec **does** have real dependencies, but it also has a couple places where it *sounds* like things should carry over and then quietly lets you “reset.” Here’s the clean dependency graph.

### The only “hard” carry-overs (must reuse outputs)

1. **Data viz → model fitting (Branin intuition check)**

* You’re expected to use the Branin heatmap + stationarity comments to sanity-check the **SE GP hyperparameters** you learn (length-scales / output-scale / mean).
* This is explicitly asked: “Do they agree with your expectations given your visualization?” 

2. **Model fitting → Bayesian optimization (model choice + transform)**

* This is the main dependency you’re worried about, and yes: it’s intended.
* You do model search/BIC + possible transforms in **Model fitting**, then in **Bayesian optimization** you “fix a choice of GP model for each dataset… selecting the best model (possibly coupled with a transformation) you found.” 
  So the BO section is supposed to inherit:
* chosen kernel/mean structure (per dataset)
* chosen output transform (if any)
* and (for *this BO part*) hyperparameters fixed to MLE on the dataset you’re using

3. **Model fitting Branin 32 Sobol points → BO Branin EI heatmaps**

* The BO section explicitly says: for Branin, make posterior mean/std heatmaps **from the 32 datapoints we used before using the model you selected**, then make an EI heatmap and mark the argmax. 
  That’s a direct “reuse that exact dataset D” dependency.

4. **BO experiment → gap / learning curves / t-tests**

* Obviously, all evaluation bullets depend on having stored the trajectories from the 20 runs (BO + random). 

---

### The “soft” carry-overs (not mandatory, but strongly implied)

5. **Data viz transforms (LDA/SVM) → model fitting transforms**

* You’re asked in data viz: “find a transformation that makes performance better behaved” 
  Then later, in model search you’re told you “may also consider transformations… to improve the bic further” 
  This implies: use the transformations you motivated earlier as candidates for BIC search, or at least explain why you did/didn’t.

6. **Snoek paper → EI implementation + interpretation**

* The EI formula reference is explicit. 
  Not a computational dependency, but a citation/justification dependency.

---

## Where it’s *easy* to think there should be carry-over, but the spec doesn’t require it

### A) The “best model” for SVM/LDA from model-fitting search vs what you use in BO runs

The spec wants you to:

* do BIC search on SVM/LDA using **32 randomly sampled observations** 
* then later “fix a choice of GP model for each dataset… selecting the best model you found” 

**But** in the BO experiment, you start each run with 5 random points and then keep adding points. The GP you fit during BO is on *that evolving dataset*, not on the original 32 used for BIC search.

So the intended carry-over is:

* **model class choice** (kernel/mean/transform), from the 32-subsample BIC search
  not
* the specific fitted hyperparameters from that one 32-subsample

In BO, you refit (or at least recondition) on the run’s current data each iteration; the “fix hyperparameters to MLE” line means: at each fit, you’re doing MLE hyperparameters (point estimate), not MCMC integration. It does **not** mean “use the exact same hyperparameter vector found in the BIC-search subsample forever.” The wording is ambiguous, but your implementation choice should be consistent and stated.

### B) The Branin “log transform” modeling results vs the earlier “don’t apply transformation” instruction

They explicitly say: first Branin modeling steps: “do not apply any transformation” 
Then they explicitly ask you to repeat with a log transform later. 
So transformation is intentionally staged: baseline → transform comparison → model search.

---

## Minimal dependency checklist (if you want your report to feel “connected”)

* **Carry over 1:** Branin heatmap intuition → comment on learned length-scales (SE GP)
* **Carry over 2:** Transform ideas (from viz) → used as candidates in model search (or rejected with reason)
* **Carry over 3:** Best model per dataset (kernel/mean/transform) from BIC search → used in BO
* **Carry over 4:** Branin’s same 32 Sobol points → used again for EI heatmap sanity check
* **Carry over 5:** BO trajectories (20 runs) → used for learning curves + mean gaps + t-tests + speedup claim

If any of those links are missing, the project can feel like disconnected mini-assignments, even if each bullet is answered.

---

## The core confusion you flagged (model fitting vs BO) in one sentence

Model fitting is used to choose **the surrogate model form** (and transformations); Bayesian optimization uses that chosen form as the engine to decide points, but the dataset and the GP posterior evolve per run.

That’s the intended dependency: **structure carries over, not the specific fitted dataset** (except the explicit Branin 32-point reuse for the heatmaps).

---

## Coherency strategy (naming + “source of truth”)

To keep the project feeling like one connected investigation (instead of three mini‑assignments), we’ll make the carryovers explicit *and* label artifacts so it’s obvious what is:

- **Required baseline** (answers a prompt bullet, but not used downstream)
- **Selected choice** (the model/transform carried into the next section)
- **Exploration** (extra experiments, not required by the bullet list)
- **Legacy/stale** (older artifacts that should not be trusted anymore)

### Source of truth

- The canonical narratives live in `data_visualization/report.md`, `model_fitting/report.md`, `bayesian_optimization/report.md`, and `bonus/report.md`.
- Any PDFs (`*_report.pdf`, `report.pdf`) are treated as *derived artifacts* and can drift; if we keep them, we should label them as `legacy_*.pdf` unless we regenerate them from the current markdown.

### Naming convention

- **Baseline (prompt-required):** prefix `baseline_` (or label in caption if renaming is deferred)
  - Example: `baseline_branin_se_*`
- **Selected (carried forward):** prefix `selected_`
  - Example: `selected_branin_surrogate_*`
- **Explorations:** live under `*/explorations/` and use `exp_` prefix when ambiguous
  - Example: `explorations/exp_branin_model_comparison.png`
- **Legacy/stale:** prefix `legacy_` (or move into `legacy/` subfolder)
  - Example: `legacy_kappa_sensitivity.png`

### Minimal “carryover headers” in each report

To make dependencies unmistakable, each section’s `report.md` should include a short block near the top:

- **Inputs from previous section** (what we are reusing)
- **Outputs carried forward** (what the next section will use)

That gives the reader a dependency chain they can follow without reverse‑engineering code.

### Current carryovers (as implemented)

- **Branin surrogate carried forward**
  - Selected in: `model_fitting/report.md` (best model section)
  - Implemented in: `bayesian_optimization/bayesian_optimization.py` and `bonus/bonus_acquisition_comparison.py`
  - Model: original scale + constant mean + **SE + Periodic(x1)**, noise σ=0.001 (in original units)
- **LDA/SVM surrogate carried forward**
  - Selected in: `model_fitting/report.md` (benchmark model search)
  - Implemented in: `bayesian_optimization/bayesian_optimization.py` and `bonus/bonus_acquisition_comparison.py`
  - Model: **Matern 3/2** + **log(y+1)** transform, noise σ=0.001 (in original units)

- **Branin “same 32 points” reuse**
  - Generated in model fitting with Sobol seed=42
  - Reused in BO EI heatmaps with the same Sobol seed=42
