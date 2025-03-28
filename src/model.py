import numpy as np
import pandas as pd


class ProjectModel:
    def __init__(self, model):
        self.model = model
        self.X_acc = None
        self.y_acc = None

    def fit(self, X, y):
        if self.X_acc is None:
            self.X_acc = X
            self.y_acc = y
        else:
            self.X_acc = np.concatenate([self.X_acc, X])
            self.y_acc = np.concatenate([self.y_acc, y])
        self.model.fit(self.X_acc, self.y_acc)

    def predict(self, X):
        return self.model.predict(X)