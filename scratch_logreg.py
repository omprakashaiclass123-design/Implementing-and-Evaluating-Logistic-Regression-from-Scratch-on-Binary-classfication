
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def train_lr(X, y, lr=0.1, epochs=2000):
    n, m = X.shape
    W = np.zeros(m)
    b = 0
    for _ in range(epochs):
        z = X@W + b
        y_pred = sigmoid(z)
        dw = (1/n)*X.T@(y_pred - y)
        db = np.mean(y_pred - y)
        W -= lr*dw
        b -= lr*db
    return W, b
