class Trainer:
    def __init__(self, model):
        self.model = model
        self.X_accumulated = []
        self.y_accumulated = []

    def train(self, X, y):
        self.X_accumulated.extend(X)
        self.y_accumulated.extend(y)
        self.model.fit(self.X_accumulated, self.y_accumulated)

    def predict(self, X):
        return self.model.predict(X)