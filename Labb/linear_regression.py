import numpy as np
from scipy import stats


class LinearRegression:
    def __init__(self, X, y):
        self.X_raw = X
        self.y = y
        ones = np.ones((X.shape[0], 1))
        self.X = np.concatenate((ones, X), axis=1)

        self.n = X.shape[0]
        self.d = X.shape[1]
        
        self.beta = None
        self.y_hat = None
        self.SSE = None
        self.SSR = None
        self.Syy = None
        self.R2 = None

    def fit(self):
        XT = self.X.T
        self.beta = np.linalg.inv(XT @ self.X) @ XT @ self.y
        self.y_hat = self.X @ self.beta
        self.SSE = np.sum((self.y - self.y_hat) ** 2)
        y_mean = np.mean(self.y)
        self.Syy = np.sum((self.y - y_mean) ** 2)
        self.SSR = self.Syy - self.SSE
        self.R2 = self.SSR / self.Syy

    def variance(self):
        return self.SSE / (self.n - self.d - 1)

    def standard_deviation(self):
        return np.sqrt(self.variance())

    def rmse(self):
        return np.sqrt(self.SSE / self.n)

    def r_squared(self):
        return self.R2

    def f_test(self):
        sigma2_hat = self.variance()
        F_stat = (self.SSR / self.d) / sigma2_hat
        p_value = stats.f.sf(F_stat, self.d, self.n - self.d - 1)
        return F_stat, p_value

    def covariance_matrix(self):
        XT_X_inv = np.linalg.inv(self.X.T @ self.X)
        return XT_X_inv * self.variance()

    def t_test(self):
        cov_matrix = self.covariance_matrix()
        se = np.sqrt(np.diag(cov_matrix))

        t_stats = self.beta / se
        df = self.n - self.d - 1

        p_values = 2 * stats.t.sf(np.abs(t_stats), df)
        return t_stats, p_values

    def confidence_intervals(self, alpha=0.05):
        cov_matrix = self.covariance_matrix()
        se = np.sqrt(np.diag(cov_matrix))
        df = self.n - self.d - 1
        t_crit = stats.t.ppf(1 - alpha/2, df)

        lower = self.beta - t_crit * se
        upper = self.beta + t_crit * se

        return np.column_stack((lower, upper))

    def pearson_matrix(self):
        d = self.X_raw.shape[1]
        corr_matrix = np.zeros((d, d))

        for i in range(d):
            for j in range(d):
                r, _ = stats.pearsonr(self.X_raw[:, i],
                                      self.X_raw[:, j])
                corr_matrix[i, j] = r

        return corr_matrix