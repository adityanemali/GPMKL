"""
 Author: Nemali Aditya <aditya.nemali@dzne.de>
==================
Multikernel Learning - Gaussian process Regression - Implementation based on
C. E. Rasmussen & C. K. I. Williams, Gaussian Processes.

For implementation, refer to examples folder.
==================
"""

import numpy as np
from scipy.optimize import minimize
from GPMKL.utils import Kernel  # Fixed import to use the correct module pathKernel


class GaussianProcessRegression:
    def __init__(self, kernel_type='rbf', **kernel_params):
        """ Initialize the regression model with a specific kernel type and parameters. """
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.kernel = Kernel.kernel(kernel_type, **kernel_params)
        self.noise = 1e-10  # Small noise term for numerical stability
        self.trained = False

    def fit(self, X_train, y_train):
        """ Fit the Gaussian Process model to the training data. """
        self.X_train = X_train
        self.y_train = y_train
        self.optimize_kernel_params()
        self.K = self.kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        self.L = np.linalg.cholesky(self.K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y_train))
        self.trained = True
        return self

    def predict(self, X_test):
        """ Predict using the Gaussian Process model. """
        if not self.trained:
            raise ValueError("The model must be fitted before prediction.")
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test) + self.noise * np.eye(len(X_test))

        K_s_T = np.linalg.solve(self.L, K_s)
        mu_s = K_s.T.dot(self.alpha)

        v = np.linalg.solve(self.L, K_s)
        var_s = np.diag(K_ss) - np.sum(v ** 2, axis=0)

        return mu_s, var_s

    def log_marginal_likelihood(self):
        """ Calculate the log marginal likelihood of the model. """
        return -0.5 * np.dot(self.y_train.T, self.alpha) - np.sum(np.log(np.diag(self.L))) - len(
            self.y_train) / 2 * np.log(2 * np.pi)

    def optimize_kernel_params(self):
        """ Optimize the kernel hyperparameters using the scipy minimize function. """

        def objective(params):
            if self.kernel_type == "rbf":
                self.kernel = Kernel.kernel(self.kernel_type, sigma_f=params[0], l=params[1])
            elif self.kernel_type == "linear":
                self.kernel = Kernel.kernel(self.kernel_type, c=params[0])
            K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
            return 0.5 * np.dot(self.y_train.T, alpha) + np.sum(np.log(np.diag(L))) + len(self.y_train) / 2 * np.log(
                2 * np.pi)

        initial_params = [self.kernel_params.get('sigma_f', 1.0),
                          self.kernel_params.get('l', 1.0)] if self.kernel_type == "rbf" else [
            self.kernel_params.get('c', 1.0)]
        bounds = [(0.1, 10), (0.1, 10)] if self.kernel_type == "rbf" else [(0.1, 10)]
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        optimized_params = result.x
        if self.kernel_type == "rbf":
            self.kernel_params = {'sigma_f': optimized_params[0], 'l': optimized_params[1]}
        elif self.kernel_type == "linear":
            self.kernel_params = {'c': optimized_params[0]}
        self.kernel = Kernel.kernel(self.kernel_type, **self.kernel_params)