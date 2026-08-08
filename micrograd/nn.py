import random
from functools import reduce
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    """ shared linear part w·x + b; subclasses apply their activation on top """

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)

    def _lin(self, x):
        return sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{type(self).__name__}({len(self.w)})"

# ReLU(x) = max(0,x)
class NeuronReLU(Neuron):

    def __call__(self, x):
        return self._lin(x).relu()

# PReLU(x) = max(0,x) + α·min(0,x), learnable negative slope α
class NeuronPReLU(Neuron):

    def __init__(self, nin):
        super().__init__(nin)
        self.alpha = Value(0.01)

    def __call__(self, x):
        act = self._lin(x)
        return act.relu() + self.alpha * (act - act.relu())

    def parameters(self):
        return super().parameters() + [self.alpha]

# Swish-β(x) = x·σ(βx), learnable β: β→0 linear, β→∞ ReLU-like
class NeuronSwish(Neuron):

    def __init__(self, nin):
        super().__init__(nin)
        self.beta = Value(1.0)

    def __call__(self, x):
        act = self._lin(x)
        return act * (self.beta * act).sigmoid()

    def parameters(self):
        return super().parameters() + [self.beta]

# PELU(x) = (a/b)·x for x≥0, a·(e^(x/b)−1) for x<0; learnable a,b (assumed >0, not enforced)
class NeuronPELU(Neuron):

    def __init__(self, nin):
        super().__init__(nin)
        self.pa, self.pb = Value(1.0), Value(1.0)

    def __call__(self, x):
        act = self._lin(x)
        r = act.relu()  # act - r = min(0, act), so each branch vanishes on the other's domain
        return (self.pa / self.pb) * r + self.pa * (((act - r) / self.pb).exp() - 1)

    def parameters(self):
        return super().parameters() + [self.pa, self.pb]

# SReLU(x): t_l + a_l·(x−t_l) below t_l, identity in between, t_r + a_r·(x−t_r) above t_r
# (assumes t_l ≤ t_r, not enforced)
class NeuronSReLU(Neuron):

    def __init__(self, nin):
        super().__init__(nin)
        self.tl, self.al = Value(-1.0), Value(0.1)
        self.tr, self.ar = Value(1.0), Value(1.0)

    def __call__(self, x):
        act = self._lin(x)
        return act + (self.ar - 1) * (act - self.tr).relu() - (self.al - 1) * (self.tl - act).relu()

    def parameters(self):
        return super().parameters() + [self.tl, self.al, self.tr, self.ar]

# APL(x) = max(0,x) + Σ_s a_s·max(0, t_s−x), S learnable hinges → arbitrary piecewise-linear shapes
class NeuronAPL(Neuron):

    def __init__(self, nin, S=2):
        super().__init__(nin)
        self.a = [Value(random.uniform(-0.5,0.5)) for _ in range(S)]
        self.t = [Value(random.uniform(-1,1)) for _ in range(S)]

    def __call__(self, x):
        act = self._lin(x)
        return act.relu() + sum(a * (t - act).relu() for a,t in zip(self.a, self.t))

    def parameters(self):
        return super().parameters() + self.a + self.t

# Maxout(x) = max_j(w_j·x + b_j) over k affine maps; no single linear part, so not a Neuron subclass
class NeuronMaxout(Module):

    def __init__(self, nin, k=3):
        self.w = [[Value(random.uniform(-1,1)) for _ in range(nin)] for _ in range(k)]
        self.b = [Value(random.uniform(-1,1)) for _ in range(k)]

    def __call__(self, x):
        acts = [sum((wi*xi for wi,xi in zip(wj, x)), bj) for wj,bj in zip(self.w, self.b)]
        return reduce(lambda m, a: m + (a - m).relu(), acts)  # max(m,a) = m + relu(a−m)

    def parameters(self):
        return [wi for wj in self.w for wi in wj] + self.b

    def __repr__(self):
        return f"NeuronMaxout({len(self.w[0])}, k={len(self.w)})"

# PAU(x) = P(x)/Q(x), P = Σ_{j≤m} p_j·x^j, Q = 1 + |Σ_{j≤n} q_j·x^(j+1)| ≥ 1 (pole-free "safe" form)
class NeuronPAU(Neuron):

    def __init__(self, nin, m=3, n=2):
        super().__init__(nin)
        # P ≈ identity at init; q must start off 0 or |·| kills its gradient
        self.p = [Value(0), Value(1)] + [Value(random.uniform(-0.1,0.1)) for _ in range(m-1)]
        self.q = [Value(random.uniform(-0.1,0.1)) for _ in range(n)]

    def __call__(self, x):
        act = self._lin(x)
        horner = lambda cs: reduce(lambda acc, c: acc * act + c, reversed(cs))
        P, z = horner(self.p), act * horner(self.q)
        return P / (1 + z.relu() + (-z).relu())  # |z| = relu(z) + relu(−z)

    def parameters(self):
        return super().parameters() + self.p + self.q

# KAF(x) = Σ_i α_i·exp(−γ(x−d_i)²): fixed Gaussian dictionary d, learnable mixing weights α
class NeuronKAF(Neuron):

    def __init__(self, nin, D=5, lo=-2.0, hi=2.0):
        super().__init__(nin)
        step = (hi - lo) / (D - 1)
        self.d = [lo + step*i for i in range(D)]
        self.gamma = 1 / (2 * step**2)
        self.alpha = [Value(random.uniform(-1,1)) for _ in range(D)]

    def __call__(self, x):
        act = self._lin(x)
        return sum(a * (-self.gamma * (act - d)**2).exp() for a,d in zip(self.alpha, self.d))

    def parameters(self):
        return super().parameters() + self.alpha

# ACON-C(x) = (p1−p2)·x·σ(β(p1−p2)x) + p2·x; β smoothly switches activation on (Swish-like) / off (linear)
class NeuronACON(Neuron):

    def __init__(self, nin):
        super().__init__(nin)
        self.p1, self.p2, self.beta = Value(1.0), Value(0.0), Value(1.0)

    def __call__(self, x):
        act = self._lin(x)
        d = (self.p1 - self.p2) * act
        return d * (self.beta * d).sigmoid() + self.p2 * act

    def parameters(self):
        return super().parameters() + [self.p1, self.p2, self.beta]

# Soft-Exponential(x): (e^(αx)−1)/α + α for α>0, x for α=0, −ln(1−α(x+α))/α for α<0
# learnable α interpolates exp ↔ identity ↔ log; log branch needs 1−α(x+α) > 0 (not enforced)
class NeuronSoftExp(Neuron):

    def __init__(self, nin):
        super().__init__(nin)
        self.alpha = Value(0.1)  # off 0: the α=0 branch gives α zero gradient (would stay stuck)

    def __call__(self, x):
        act, a = self._lin(x), self.alpha
        if a.data > 0:
            return ((a * act).exp() - 1) / a + a
        if a.data < 0:
            return -(1 - a * (act + a)).log() / a
        return act

    def parameters(self):
        return super().parameters() + [self.alpha]

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
