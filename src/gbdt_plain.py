"""Plain gradient boosting in under 40 lines (core loop), Friedman 2001 style.

F_0 = mean(y); r = y - F; fit tree to r; F += lr * tree(x); repeat.
Squared loss => negative gradient == residual. Shrinkage lr=0.1 per video.
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class PlainGBDT:
    def __init__(self, n_estimators=300, max_depth=3, learning_rate=0.1, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.init_ = 0.0
        self.trees_ = []

    def fit(self, X, y):
        self.init_ = float(np.mean(y))
        F = np.full_like(y, self.init_, dtype=float)
        self.trees_ = []
        for m in range(self.n_estimators):
            r = y - F  # neg gradient of (1/2)(y-F)^2
            t = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.seed + m)
            t.fit(X, r)
            F = F + self.learning_rate * t.predict(X)
            self.trees_.append(t)
        return self

    def predict(self, X):
        F = np.full(X.shape[0], self.init_, dtype=float)
        for t in self.trees_:
            F = F + self.learning_rate * t.predict(X)
        return F

    def staged_rmse(self, X, y):
        from sklearn.metrics import mean_squared_error
        F = np.full(X.shape[0], self.init_, dtype=float)
        out = [float(mean_squared_error(y, F) ** 0.5)]
        for t in self.trees_:
            F = F + self.learning_rate * t.predict(X)
            out.append(float(mean_squared_error(y, F) ** 0.5))
        return out
