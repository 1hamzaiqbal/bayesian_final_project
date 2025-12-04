The project will be an investigation into applying Bayesian optimization to automatic and
efficient hyperparameter tuning of machine learning models.
We will use two sources of data for this investigation. The first is a classical synthetic bench-
mark function used in optimization, the Branin (sometimes Branin–Hoo) function. See here for a
description, formula, and some implementations:
https://www.sfu.ca/~ssurjano/branin.html
This is a function of a two-dimensional input and can be useful for visualizing the behavior of the
methods you will implement. The goal is to minimize the function over the specified box-bounded
domain.
The second is real data from a hyperparameter tuning task on two different models: svm and
online lda. These datasets were used in the paper “Practical Bayesian Optimization of Machine
Learning Algorithms” by Snoek, et al. The paper is available at
http://tiny.cc/bopt
You should read it!
Given some data, the performance of each methods was evaluated on a three-dimensional grid
of hyperparameter values, defining an objective function f (θ), where θ is a three-dimensional
hyperparameter vector. Our goal is to maximize the performance of the model as a function of
the hyperparameters. As the cost of training and evaluating the performance of these models is so
great, we will use the precomputed grids instead. The data is available here:
https://github.com/mwhoffman/benchfunk/tree/master/benchfunk/functions/data
For each problem, the goal is to minimize the value in the fourth column given the values of the
hyperparameters in the first three. You can ignore the final column.
Note that all of these are minimization problems.
The project will comprise a series of smaller components that you must complete. You will then
compile your findings into a report. I expect every bulleted item to be addressed in the report. The
report will be due Friday, 12 December 2025, the last day of class.
If you have any question about what may or may not be appropriate for any of these parts,
please feel free to ask on Slack!
Data visualization
First let’s do some visualization of the data to get some insight into the functions we are dealing
with.
• Make a heatmap of the value of the Branin function over the domain X = [−5, 10] × [0, 15]
using a dense grid of values, with 1000 values per dimension, forming a 1000 × 1000 image.
• Describe the behavior of the function. Does it appear stationary? (That is, does the behavior
of the function appear to be relatively constant throughout the domain?)
• Can you find a transformation of the data that makes it more stationary?
3
• Make a kernel density estimate of the distribution of the values for the lda and svm bench-
marks. Interpret the distributions.
• Again, can you find a transformation that makes the performance better behaved?