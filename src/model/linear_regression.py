class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X_b = self._add_bias(X)
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        X_b = self._add_bias(X)
        return X_b.dot(np.concatenate(([self.intercept], self.coefficients)))

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]