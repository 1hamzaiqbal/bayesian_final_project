Model fitting
Before we can succeed with optimization, we must first determine whether we can build a useful
model of the objective function. If we can’t find an informative model that can make reasonable
predictions, it would be foolish to try to use the model to make decisions.
We begin with the Branin function where we can make useful diagnostic plots. For this first
series of steps, do not apply any transformation to the outputs of the function; use its output directly.
• Select a set of 32 training points for the Branin function in the domain X = [−5, 10] ×
[0, 15] using a Sobol sequence. This is a so-called low-discrepancy sequence that produces
“quasirandom” points that fill the space relatively evenly; see the Wikipedia article for more
information, and note that this should be built into most scientific software. Measure the
function at these locations to form a dataset D.
• Fit a Gaussian process model to the data using a constant mean and a squared exponential
covariance. As the output of a deterministic computer program, there is no “noise” in our
observations, so we should fix the standard deviation of the noise to a small value such as
0.001. This is possible in most gp software packages.
Maximize the marginal likelihood of the data as a function of the hyperparameters: constant
mean value and the length scale and output scale of the covariance function.
• What values did you learn for the hyperparameters? Do they agree with your expectations
given your visualization?
• Make a heatmap of the Gaussian process posterior mean as you did of the function. Compare
the predicted values with the true values. Do you see systematic errors?
• Make a heatmap of the Gaussian process posterior standard deviation (not variance!) as you
did of the function. Do the values make sense? Does the scale make sense? Does the standard
deviation drop to near zero at your data points?
Note that if these heatmaps do not agree with your intuition, it may indicate a bug or mistake
somewhere!
• Make a kernel density estimate of the z-scores of the residuals between the posterior mean of
trained Gaussian process and the true values. If the Gaussian process model is well calibrated
this should be approximately standard normal. Is that the case?
• Repeat the above using a log transformation to the output of the Branin function. Does the
marginal likelihood improve? Does the model appear better calibrated?
Now let’s try some different models. Given a gp mean function and covariance, we can derive a
score for how well those choices fit a given dataset, such as the dataset of 32 observations D above.
The most natrual score would be the model evidence after integrating over the hyperparameters,
but unfortunately this is intractable for Gaussian process models. A widespread and convenient
approximation is called the Bayesian information criterion (bic). Look up the Wikipedia article and
4
read it! To compute the bic we first find the values of the hyperparameters maximizing the (log)
marginal likelihood:
ˆθ = arg max
θ
log p(y | X, θ).
Now the bic is a combination of two terms rewarding model fit and penalizing model complexity:
bic = |θ| log|D| − 2 log p(y | X, ˆθ),
where |θ| is the total number of hyperparameters (above, 3) and |D| is the number of observations
(above, 32). Given a set of models, we prefer the one with the lowest bic, and this score can actually
be interpreted as an approximation to the (negative) log model evidence.
• Compute the bic score for the data and model from the last part.
• Considering bic as a function of the choice of mean and covariance functions (µ, K), as
well as the dataset described above, attempt a search over models to find the best possible
explanation of the data. What is the best model you found and its bic score? The notion of
the “compositional kernel grammar” may be useful for automating the search, or you could
do it by hand. You may also consider transformations of the data to improve the bic further.
• Perform a similar search for the svm and lda datasets, using 32 randomly sampled observations
from each dataset. What is the best gp model you found for each?