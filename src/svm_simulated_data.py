import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.svm

# Create a random dataframe and standardize it.
np.random.seed(100)
data = pd.DataFrame(np.random.randint(0,100,size=(8000, 1000)))
data_test = pd.DataFrame(np.random.randint(0,2,size=(8000, 1)))

x = np.asarray(data)[:, 0:-1]
y = np.asarray(data)[:, -1]*2 - 1
data_test = np.array(data_test).T[0]

# Divide the data into train, test sets
x_train = x[data_test == 0, :]
x_test = x[data_test == 1, :]
y_train = y[data_test == 0]
y_test = y[data_test == 1]

# Standardize the data.
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_train = len(y_train)
n_test = len(y_test)
d = np.size(x, 1)


#  Implement fast gradient to train linear SVM
def grad_obj(beta, lambduh, x , y ):
    """
    This function calculated the gradient of the Linear Support
    Vector Machine with square Hinge Loss
    """
    yx = y[:, np.newaxis]*x
    if np.all(np.zeros_like(y) > 1-y*np.dot(x, beta)):
        return 2*lambduh*beta
    else:
        return -2/np.size(x, 0)*np.sum(y[:, np.newaxis]*x*(1-y*np.dot(x, beta))[:, np.newaxis], axis=0)+2*lambduh*beta

def objective(beta, lambduh, x, y):
    """
    This function calculates the objective function of
    Linear Support Vector Machine
    with square Hinge loss
    """
    if np.all(np.zeros_like(y) > 1-y*np.dot(x, beta)):
        return lambduh * np.linalg.norm(beta)**2
    else:
        return 1/len(y) * np.sum((1-y*np.dot(x, beta))**2) + lambduh * np.linalg.norm(beta)**2

def backtrack(beta, lambduh, eta=1, alpha=0.5, betaparam=0.8,maxiter=100, x = x_train, y= y_train ):
    """
    This function implements the backtracking rule
    """
    grad_beta = grad_obj(beta, lambduh, x=x, y=y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_eta = 0
    iter = 0
    while found_eta == 0 and iter < maxiter:
        if objective(beta - eta * grad_beta, lambduh, x=x, y=y) < objective(beta, lambduh, x=x, y=y) \
                - alpha * eta * norm_grad_beta ** 2:
            found_eta = 1
        elif iter == maxiter - 1:
            break
        else:
            eta *= betaparam
            iter += 1
    return eta


def fastgradient(beta_init, theta_init, lambduh, eta_init, maxiter, x= x_train, y= y_train):
    """
    This function implements fast gradient algorithm with backtracking rule
    to tune the step size.
    It calls backtrack and gradient functions.
    """
    beta = beta_init
    theta = theta_init
    grad_theta = grad_obj(theta, lambduh, x=x, y=y)
    beta_vals = beta
    theta_vals = theta
    iterate = 0
    while iterate < maxiter:
        eta = backtrack(theta, lambduh, eta=eta_init, x=x, y=y)
        beta_new = theta - eta*grad_theta
        theta = beta_new + iterate/(iterate+3)*(beta_new-beta)
        beta_vals = np.vstack((beta_vals, beta_new))
        theta_vals = np.vstack((theta_vals, theta))
        grad_theta = grad_obj(theta, lambduh, x=x, y=y)
        beta = beta_new
        iterate += 1
    return beta_vals, theta_vals


def misclass_error(beta, x = x_test, y= y_test):
    """
    This function calculates the mis-classification error between
    the testing y and predicted y.
    """
    y_pred = (np.dot(x, beta) > 0)*2 - 1
    return np.mean(y_pred != y)


def objective_plot(betas_fg, lambduh, x= x_train, y= y_train, save_file=''):
    num_points = np.size(betas_fg, 0)
    objs_fg = np.zeros(num_points)
    for i in range(0, num_points):
        objs_fg[i] = objective(betas_fg[i, :], lambduh, x=x, y=y)
    fig, ax = plt.subplots()
    ax.plot(range(1, num_points + 1), objs_fg)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.title('Objective value vs. iteration when lambda='+str(lambduh))
    if not save_file:
        plt.show()
    else:
        plt.savefig(save_file)


if __name__ == '__main__':

    ## Initiate lambda = 1
    lambduh_in = 1
    beta_in = np.zeros(d)
    theta_in = np.zeros(d)
    eta_init = 1/(scipy.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals=(d-1, d-1), eigvals_only=True)[0]+lambduh_in)
    maxiter = 100
    betas_fastgrad, thetas_fastgrad = fastgradient(beta_in, theta_in, lambduh_in, eta_init, maxiter)
    print('Optimal betas:', betas_fastgrad[-1, :])
    objective_plot(betas_fastgrad, lambduh_in, save_file='')
    print('Misclassification error when lambda=1:', misclass_error(betas_fastgrad[-1, :]))
