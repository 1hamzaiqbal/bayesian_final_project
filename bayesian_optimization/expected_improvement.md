Probability of Improvement. One intuitive strategy is to maximize the probability of improving over the best current value (Kushner, 1964). Under the GP this can be computed
analytically as
aPI(x ; {xn, yn}, θ) = Φ(γ(x)) γ(x) = f(xbest) − µ(x ; {xn, yn}, θ)
σ(x ; {xn, yn}, θ)
(1) .
Expected Improvement. Alternatively, one could choose to maximize the expected improvement (EI) over the current best. This also has closed form under the Gaussian process:
(2) aEI(x ; {xn, yn}, θ) = σ(x ; {xn, yn}, θ) (γ(x) Φ(γ(x)) + N (γ(x) ; 0, 1))
GP Upper Confidence Bound. A more recent development is the idea of exploiting lower
confidence bounds (upper, when considering maximization) to construct acquisition functions
that minimize regret over the course of their optimization (Srinivas et al., 2010). These acquisition functions have the form
(3) aLCB(x ; {xn, yn}, θ) = µ(x ; {xn, yn}, θ) − κ σ(x ; {xn, yn}, θ),
with a tunable κ to balance exploitation against exploration