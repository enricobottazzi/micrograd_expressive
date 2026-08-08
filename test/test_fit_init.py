import random
import numpy as np
import matplotlib.pyplot as plt

from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP
from micrograd.utils import loss, plot

def test_fit_relu():
    np.random.seed(1337)
    random.seed(1337)

    # relu(2x+7) dataset, domain [-10, 10]
    X = np.linspace(-10, 10, 10000).reshape(-1, 1)
    y = np.maximum(0, 2 * X + 7)

    # train-test split
    ri = np.random.permutation(len(X))
    n_train = int(0.8 * len(X))
    X_train, y_train = X[ri[:n_train]], y[ri[:n_train]]
    X_test,  y_test  = X[ri[n_train:]], y[ri[n_train:]]

    plot(X_train, y_train, 'relu_target.png')
    
    # initialize a model (single neuron)
    model = Neuron(1)
    print(model)
    print("number of parameters", len(model.parameters()))

    # training loop
    for k in range(1000):
        # forward (batched train)
        total_loss = loss(model, X_train, y_train, batch_size=32)

        # backward
        model.zero_grad()
        total_loss.backward()

        # update (sgd)
        learning_rate = 0.01
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 100 == 0:
            train_loss = loss(model, X_train, y_train)
            test_loss = loss(model, X_test, y_test)
            print(f"step {k} train {train_loss.data:.4f} test {test_loss.data:.4f} w={model.w[0].data:.4f} b={model.b.data:.4f}")

    train_loss = loss(model, X_train, y_train)
    test_loss = loss(model, X_test, y_test)
    print(f"final train {train_loss.data:.4f} test {test_loss.data:.4f} w={model.w[0].data:.4f} b={model.b.data:.4f}")
    preds = [model(list(map(Value, xrow))).data for xrow in X_train]
    plot(X_train, preds, 'relu_preds.png')

def test_fit_leaky_relu():
    np.random.seed(1337)
    random.seed(1337)

    # leaky_relu(2x+7, neg_slope=0.5) dataset, domain [-10, 10]
    X = np.linspace(-10, 10, 10000).reshape(-1, 1)
    t = 2 * X + 7
    y = np.where(t > 0, t, 0.5 * t)

    # train-test split
    ri = np.random.permutation(len(X))
    n_train = int(0.8 * len(X))
    X_train, y_train = X[ri[:n_train]], y[ri[:n_train]]
    X_test,  y_test  = X[ri[n_train:]], y[ri[n_train:]]

    plot(X_train, y_train, 'leaky_relu_target.png')

    # initialize a model (single neuron)
    model = Neuron(1)
    print(model)
    print("number of parameters", len(model.parameters()))

    # training loop
    for k in range(1000):
        # forward (batched train)
        total_loss = loss(model, X_train, y_train, batch_size=32)

        # backward
        model.zero_grad()
        total_loss.backward()

        # update (sgd)
        learning_rate = 0.01
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 100 == 0:
            train_loss = loss(model, X_train, y_train)
            test_loss = loss(model, X_test, y_test)
            print(f"step {k} train {train_loss.data:.4f} test {test_loss.data:.4f} w={model.w[0].data:.4f} b={model.b.data:.4f}")

    train_loss = loss(model, X_train, y_train)
    test_loss = loss(model, X_test, y_test)
    print(f"final train {train_loss.data:.4f} test {test_loss.data:.4f} w={model.w[0].data:.4f} b={model.b.data:.4f}")
    preds = [model(list(map(Value, xrow))).data for xrow in X_train]
    plot(X_train, preds, 'leaky_relu_preds.png')


# PReLU(x) = max(0,x) + α·min(0,x)
class NeuronWithPReLU(Neuron):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.alpha = Value(0.01)  # learnable negative slope
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if not self.nonlin:
            return act
        return act.relu() + self.alpha * (act - act.relu())

    def parameters(self):
        return self.w + [self.b, self.alpha]

    def __repr__(self):
        return f"{'PReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


def test_fit_relu_with_prelu():
    np.random.seed(1337)
    random.seed(1337)

    X = np.linspace(-10, 10, 10000).reshape(-1, 1)
    y = np.maximum(0, 2 * X + 7)

    ri = np.random.permutation(len(X))
    n_train = int(0.8 * len(X))
    X_train, y_train = X[ri[:n_train]], y[ri[:n_train]]
    X_test,  y_test  = X[ri[n_train:]], y[ri[n_train:]]

    plot(X_train, y_train, 'relu_with_p_relu_target.png')

    model = NeuronWithPReLU(1)
    print(model)
    print("number of parameters", len(model.parameters()))

    for k in range(1000):
        total_loss = loss(model, X_train, y_train, batch_size=32)
        model.zero_grad()
        total_loss.backward()
        learning_rate = 0.01
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        if k % 100 == 0:
            train_loss = loss(model, X_train, y_train)
            test_loss = loss(model, X_test, y_test)
            print(f"step {k} train {train_loss.data:.4f} test {test_loss.data:.4f} w={model.w[0].data:.4f} b={model.b.data:.4f} alpha={model.alpha.data:.4f}")

    train_loss = loss(model, X_train, y_train)
    test_loss = loss(model, X_test, y_test)
    print(f"final train {train_loss.data:.4f} test {test_loss.data:.4f} w={model.w[0].data:.4f} b={model.b.data:.4f} alpha={model.alpha.data:.4f}")
    preds = [model(list(map(Value, xrow))).data for xrow in X_train]
    plot(X_train, preds, 'relu_with_p_relu_preds.png')


def test_fit_leaky_relu_with_prelu():
    np.random.seed(1337)
    random.seed(1337)

    X = np.linspace(-10, 10, 10000).reshape(-1, 1)
    t = 2 * X + 7
    y = np.where(t > 0, t, 0.5 * t)

    ri = np.random.permutation(len(X))
    n_train = int(0.8 * len(X))
    X_train, y_train = X[ri[:n_train]], y[ri[:n_train]]
    X_test,  y_test  = X[ri[n_train:]], y[ri[n_train:]]

    plot(X_train, y_train, 'leaky_relu_with_p_relu_target.png')

    model = NeuronWithPReLU(1)
    print(model)
    print("number of parameters", len(model.parameters()))

    for k in range(1000):
        total_loss = loss(model, X_train, y_train, batch_size=32)
        model.zero_grad()
        total_loss.backward()
        learning_rate = 0.01
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        if k % 100 == 0:
            train_loss = loss(model, X_train, y_train)
            test_loss = loss(model, X_test, y_test)
            print(f"step {k} train {train_loss.data:.4f} test {test_loss.data:.4f} w={model.w[0].data:.4f} b={model.b.data:.4f} alpha={model.alpha.data:.4f}")

    train_loss = loss(model, X_train, y_train)
    test_loss = loss(model, X_test, y_test)
    print(f"final train {train_loss.data:.4f} test {test_loss.data:.4f} w={model.w[0].data:.4f} b={model.b.data:.4f} alpha={model.alpha.data:.4f}")
    preds = [model(list(map(Value, xrow))).data for xrow in X_train]
    plot(X_train, preds, 'leaky_relu_with_p_relu_preds.png')