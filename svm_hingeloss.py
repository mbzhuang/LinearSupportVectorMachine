"""
Before running the code in this file, load your training and testing data set as 
x_train: training predictior data set
y_train: training response data set
x_test: testing predictior data set
y_test: testing response data set
"""

import pandas as pd
import numpy as np
import scipy as sp
import sklearn
import matplotlib.pyplot as plt
import sklearn.preprocessing
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# Define all functions
def objective(beta, lambduh, xy):
    """
    Compute the value of objective function
    :param beta: coefficients
    :param lambduh: Regularization parameter
    :param xy: Precomputed values \sum x_{i1}y_i, ..., \sum x_{id}y_i
    :return: Value of objective function
    """   
    n = np.size(xy, 1)
    return np.linalg.norm(np.maximum(0, 1- np.dot(xy.T, beta)))**2/n + lambduh * np.linalg.norm(beta)**2

def computegrad(beta, lambduh, xy):
    """
    Compute the value of gradient function
    :param beta: coefficients
    :param lambduh: regularization parameter
    :param xy: precomputed values \sum x_{i1}y_i, ..., \sum x_{id}y_i
    :return: value of gradient function
    """
    n = np.size(xy, 1)
    return 2 * ((-1/n) * np.dot(xy, np.maximum(0, 1- np.dot(xy.T, beta))) + lambduh * beta)

def bt_line_search(beta, lambduh, xy, eta, alpha=0.5, betaparam=0.8, maxiter=100):
    """
    Backtracking line search 
    :param beta: coefficients
    :param lambduh: regularization parameter
    :param xy: precomputed values \sum x_{i1}y_i, ..., \sum x_{id}y_i
    :param eta: step size
    :param alpha: constant used to define sufficient decrease condition
    :param betaparam: fraction by which we decrease step size if the previous step size doesn't work
    :param maxiter: maximum iteration times
    :return: eta    
    """
    grad_beta = computegrad(beta, lambduh, xy)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_eta = 0
    iter = 0
    while found_eta == 0 and iter < maxiter:
        if objective(beta - eta * grad_beta, lambduh, xy) < objective(beta, lambduh, xy)                 - alpha * eta * norm_grad_beta ** 2:
            found_eta = 1
        else:
            eta *= betaparam
            iter += 1
    return eta

def mylinearsvm(beta_init, theta_init, lambduh, eta_init, maxiter, x=x_train, y=y_train):
    """
    Backtracking line search 
    :param beta_init: starting coefficients
    :param theta_init: starting coefficients
    :param lambduh: rgularization parameter
    :param eta_init: initial step size
    :param maxiter: maximum iteration times
    :param x: training predictior data set
    :param y: training response data set
    :return: array of all values of beta  
    """
    xy = np.dot(x.T, np.diag(y))
    beta = beta_init
    theta = theta_init
    grad_theta = computegrad(theta, lambduh, xy)
    beta_vals = beta
    theta_vals = theta
    iter = 0
    while iter < maxiter:
        t0 = time.time()
        eta = bt_line_search(theta, lambduh, xy, eta_init)
        beta_new = theta - eta*grad_theta
        theta = beta_new + iter/(iter+3)*(beta_new-beta)
        # Store all of the places we step to
        beta_vals = np.vstack((beta_vals, beta_new))
        theta_vals = np.vstack((theta_vals, theta))
        grad_theta = computegrad(theta, lambduh, xy)
        beta = beta_new
        iter += 1
    return beta_vals

def objective_plot(betas, lambduh, x=x_train, y=y_train):
    """
    Plot objective/loss function value over number of iterations 
    :param beta: coefficients
    :param lambduh: regularization parameter
    :param x: training predictior data set
    :param y: training response data set
    :param maxiter: maximum iteration times
    :return: plot 
    """
    v = np.dot(x.T, np.diag(y))
    num_points = len(betas)
    objs = np.zeros(num_points)
    for i in range(0, num_points):
        objs[i] = objective(betas[i], lambduh, v)
    plt.plot(range(1, num_points + 1), objs)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.title('Objective value vs. iteration when lambda='+str(lambduh))
    plt.show()
    
def misclassification_error(lambduh, beta, x, y):
    """
    Calculate and print the missiclassification error
    :param beta: coefficients
    :param lambduh: regularization parameter
    :param x: testing predictior data set
    :param y: testing response data set
    :return: missiclassification error
    """
    print("Misclassification Error is: " + str(np.sum(x.dot(beta)*y < 0)/len(y)) + ', when lambda is ' + str(lambduh)) 
    return np.sum(x.dot(beta)*y < 0)/len(y)

def validation(fun, lambduh_tuning, beta_init, theta_init, x_train, x_test, y_train, y_test, max_iter):
    """
    Plot log of missclassification error over minus log of lambduh 
    :param fun: the model function that return an array of all values of beta 
    :param lambduh_tuning: the rgularization parameter going to be tried
    :param theta_init: 
    :param x_train: training predictior data set
    :param y_train: training response data set
    :param x_test: testing predictior data set
    :param y_test: testing response data set
    :param maxiter: maximum iteration times
    :return: plot  
    """    
    n = np.size(lambduh_tuning, 0)
    mse = np.zeros(n) 
    eta = sp.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals = (d-1, d-1), eigvals_only=True)[0]  
    for i in range(0, n):
        eta_init = 1/(eta + lambduh_tuning[i])
        betas = fun(beta_init, theta_init, lambduh_tuning[i], eta_init, max_iter, x=x_train, y=y_train)
        beta = betas[-1]
        mse[i] = misclassification_error(lambduh_tuning[i], beta, x_test, y_test)           
    plt.figure()
    plt.plot(-np.log10(lambduh_tuning), np.log(mse))
    plt.ylabel('Log of MSE on Validation Data')
    plt.xlabel('-log(Lambda)')
    plt.title('Misclassification Error vs. Lambda')
    plt.show()

# Give intial values for all parameters, use mylinearsvm to train a model on the training data, 
# and plot the value of loss function over iterations.
beta_init = np.zeros(d)
theta_init = np.zeros(d)
lambduh = 1
eta_init = 1/(sp.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals = (d-1, d-1), eigvals_only=True)[0] + lambduh)

betas = mylinearsvm(beta_init, theta_init, lambduh, eta_init, 100)
objective_plot(betas, lambduh)
misclassification_error(lambduh, betas[-1], x_test, y_test)

# Run cross-validation to find the optimal value of lambduh. 
# Report your misclassification error for that value of lambduh.
lambduh_tuning = [10**x for x in range(-10, 10, 1)]
beta_init = np.zeros(d)
theta_init = np.zeros(d)
max_iter = 100

validation(mylinearsvm, lambduh_tuning, beta_init, theta_init, x_train, x_test, y_train, y_test, max_iter)

beta_init = np.zeros(d)
theta_init = np.zeros(d)
lambduh = 10**-3
eta_init = 1/(sp.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals = (d-1, d-1), eigvals_only=True)[0] + lambduh)

betas = mylinearsvm(beta_init, theta_init, lambduh, eta_init, 100)
objective_plot(betas, lambduh)
misclassification_error(lambduh, betas[-1], x_test, y_test)

# Now use scikit-learn’s LinearSVC package.
svc = LinearSVC(penalty='l2', loss='squared_hinge', C=1.0, fit_intercept=False, max_iter=100)
parameters = {'penalty':('l2'), 'loss':('squared_hinge'), 'C':lambduh_tuning, 'fit_intercept':['False'], 'max_iter':('100')}
clf = GridSearchCV(svc, parameters)
clf.fit(x_train, y_train)
clf.best_params_

## See if the best lambduh tuning through grid search on LinearSVC package is the same with what you found in the previous step.

