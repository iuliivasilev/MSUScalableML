import numpy as np
import pandas as pd


class Trainer:
    def __init__(self, model):
        self.model = model
        self.X_acc = []
        self.y_acc = []

    def train(self, X, y):
        self.X_acc = pd.concat([self.X_acc, X])
        self.y_acc = np.concat([self.y_acc, y])
        self.model.fit(self.X_acc, self.y_acc)

    def predict(self, X):
        return self.model.predict(X)