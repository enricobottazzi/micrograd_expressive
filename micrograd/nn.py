import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

# ReLU(x) = max(0,x)
class NeuronReLU(Module):

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"ReLUNeuron({len(self.w)})"

# PReLU(x) = max(0,x) + α·min(0,x)
class NeuronPReLU(Module):

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.alpha = Value(0.01)  # learnable negative slope

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() + self.alpha * (act - act.relu())

    def parameters(self):
        return self.w + [self.b, self.alpha]

    def __repr__(self):
        return f"PReLUNeuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, neuron):
        self.neurons = [neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, neuron):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], neuron) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
