import numpy as np
#from scipy.stats import st

class LinearRegression:
    def __init__(self, X, y):
        self.y = y
        ones = np.ones((X.shape[0], 1))
        self.X = np.concatenate((ones, X), axis=1)

        self.n = X.shape[0]
        self.d = X.shape[1]

        self.beta = None
        self.y_hat = None
        self.SSE = None

    def fit(self):
        XT = self.X.T
        self.beta = np.linalg.inv(XT @ self.X) @ XT @ self.y
        self.y_hat = self.X @ self.beta
        self.SSE = np.sum((self.y - self.y_hat) ** 2)

    def variance(self):
        return self.SSE / (self.n - self.d - 1)

    def standard_deviation(self):
        return np.sqrt(self.variance())

    def rmse(self):
        return np.sqrt(self.SSE / self.n)