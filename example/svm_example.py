"""
This module is to compare between the LinearSVM algorithm with sklearn
"""
import sklearn.preprocessing
import sklearn.svm

import src.svm_real_dataset


## From algorithm
lambduh_in = 1
beta_in = np.zeros(d)
theta_in = np.zeros(d)
eta_init = 1/(scipy.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals=(d-1, d-1), eigvals_only=True)[0]+lambduh_in)
maxiter = 100
betas_fastgrad, thetas_fastgrad = fastgradient(beta_in, theta_in, lambduh_in, eta_init, maxiter)
print('Optimal betas:', betas_fastgrad[-1, :])
objective_plot(betas_fastgrad, lambduh_in, save_file='')
print('Misclassification error when lambda=1:', misclass_error(betas_fastgrad[-1, :]))

## compare with sklearn
linear_svc = sklearn.svm.LinearSVC(penalty='l2', C=1/(2*lambduh_in*n_train), fit_intercept=False, tol=10e-8, max_iter=1000)
linear_svc.fit(x_train, y_train)
print('Estimated beta from sklearn:', linear_svc.coef_)
print('Estimated beta from code:', betas_fastgrad[-1, :])

print('Objective value at optimum beta from sklearn:', objective(betas_fastgrad[-1, :], lambduh_in, x = x_train, y = y_train))
print('Objective value at optimum beta from code:', objective(linear_svc.coef_.flatten(), lambduh_in, x=x_train, y=y_train))
