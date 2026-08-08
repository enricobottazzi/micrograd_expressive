import numpy as np
from micrograd.engine import Value

def loss(model, X, y, batch_size=None):
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    # mse
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    scores = list(map(model, inputs))
    losses = [(si - yi)**2 for si, yi in zip(scores, yb.flatten())]
    return sum(losses) * (1.0 / len(losses))
