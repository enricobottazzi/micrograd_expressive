# Prince Chapter 3 Model

For input $\mathbf{x}\in\mathbb{R}^D$, $H$ hidden units, activation $a$, and scalar output:

$$
h_i=a(\theta_{i0}+\boldsymbol{\theta}_i^\top\mathbf{x}),
\qquad
y=\phi_0+\sum_{i=1}^{H}\phi_i h_i.
$$

The hidden layer is nonlinear; the output layer is affine with no activation.

## Construction

```python
from micrograd.nn import Layer, Module, NeuronLinear

class PrinceModel(Module):
    def __init__(self, input_size, hidden_units, activation):
        self.layers = [
            Layer(input_size, hidden_units, activation),
            Layer(hidden_units, 1, NeuronLinear),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

Example: `PrinceModel(1, 3, NeuronReLU)`.

## Parameter Count

If each activation has $Q_a$ learnable parameters per hidden unit:

$$
P=H(D+1)+HQ_a+(H+1)=H(D+Q_a+2)+1.
$$

- ReLU: $Q_a=0$
- PReLU, Swish, SoftExp: $Q_a=1$
- PELU: $Q_a=2$
- ACON: $Q_a=3$
- SReLU: $Q_a=4$
- APL: $Q_a=2S$
- PAU: $Q_a=m+n+1$
- KAF: $Q_a=D_{\text{dictionary}}$

Maxout is exceptional: with $K$ affine branches,

$$
P=HK(D+1)+(H+1).
$$
