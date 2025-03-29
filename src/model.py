import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


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
    
    def score(self):
        if self.X_acc is None:
            return None
        y_pred = self.predict(self.X_acc)
        print(y_pred, self.y_acc)
        return r2_score(self.y_acc, y_pred)