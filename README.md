# LinearSVM
*Support Vector Machines* are usually Supervised models, with algorithms to implement Classification and Regression analysis.
This code implements fast gradient algorithm on Linear Support Vector Machine with squared hinge loss.

The data_test should contain either +1 or -1, so we convert the depen

## Functions
The `*objective*` function calculates the objective of the SVM with squared hinge loss, while *grad_obj* calculated the gradient.
The `*backtrack*` function implements the backtracking rule.
The `*fastgradient*` us the main function that implements the Linear Support Vector Machine.
*misclass_error* function calculates the Mis-classification error on test set.

Users need to install the following packages, if not already done:
`*numpy*
*pandas*
*sklearn*
*matplotlib*`

## Data used
There are two datasets used. 
One of them is the Spam dataset, that has been downloaded directly from https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data
It gives information whether an email is spam or not. 
Other is a simulated dataset using `np.random.randint`
