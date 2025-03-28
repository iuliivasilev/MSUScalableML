import numpy as np

def generate_data(num_samples=1000, num_features=10):
    X = np.random.rand(num_samples, num_features)
    y = np.dot(X, np.random.rand(num_features)) + np.random.normal(0, 0.1, num_samples)
    
    return X, y