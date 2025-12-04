Bayesian optimization
Let’s fix a choice of Gaussian process model for each of the datasets above by selecting the best
model (possibly coupled with a transformation) you found, and for now let’s fix the hyperparameters
of that model to be those that maximize the marginal likelihood of the dataset. We will now
implement a basic Bayesian optimization procedure and observe its performance.
• Implement the expected improvement acquisition function (formula in the Snoek, et al. paper
above), which provides a score to each possible observation location in light of the observed
data. The formula is a function of the posterior predictive mean and pointwise variance of a
point, which should be readily available.
Be careful as different authors define ei for minimization or for maximization. We have
minimization problems but we can make them maximization problems by negating if needed.
• For the Branin function, make new heatmaps for the posterior mean and standard deviation
from the 32 datapoints we used before using the model you selected. Make another heatmap
for the ei value, and place a mark where it is maximized. Does the identified point seem like
a good next observation location?
• For the Branin, svm, and lda functions, implement the following experiment:
– select 5 randomly located initial observations, D
– repeat the following 30 times:
∗ Find the point x that maximizes the expected improvement acquisition function
given the current data. For the Branin function, you can simply measure ei on a
dense grid or on a dense Sobol set and find the maximum on that dense set. For the
other datasets, you can measure ei on the unlabeled points and maximize.
∗ Add the observation
x, f(x)

to your dataset.
5
– return the final dataset (containing 35 observations)
• We will evaluate performance as follows. Using the final returned dataset, identify the best
point found. We will score optimization performance using the “gap” measure, which for
maximization is:
gap =
f(best found) − f(best initial)
f(maximum) − f(best initial)
.
You can interpret gap is the portion of the “gap” between the best initial observation among
the 5 initial data points and the optimum that we managed to traverse over the course of
optimization; this gives us a normalized score between 0 and 1 for how well we did.
• Now perform 20 runs of the above Bayesian optimization experiment using different random
initializations, and store the entire sequence of observations for each run.
For a baseline, we will use random search. Using the same initializations, implement random
search (i.e., a policy that samples an unlabeled point in the domain completely at random).
Allow random search to have a total budget of 150 observations rather than 30 (5 times the
budget), but store the entire sequence of data. This will allow us to compare with naïve
“parallel random search” below.
• Make a plot of “learning curves” for each of the methods on each of the datasets. For now
only using the first 30 observations for random search, plot the average gap achieved by
Bayesian optimization and random search as a function of the number of observations for
each of the datasets. (That is, after 1, 2, . . . 30 observations, we visualize the performance
of each method.) You’ll want to make one plot per dataset with a curve for each method,
allowing us to see how well each method compares on that dataset.
• What is the mean gap for ei and for random search using 30 observations? What about 60?
90? 120? 150? Perform a paired t-test (gasp!) comparing the performance of random search
(using 30 observations) and ei. What is the p-value? How many observations does random
search need before the p value raises above 0.05? We can interpret the result in terms of a
“speedup” of Bayesian optimization over random search.