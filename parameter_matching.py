import ast
import json

from micrograd.nn import (
    Layer, Module, NeuronACON, NeuronAPL, NeuronKAF, NeuronLinear,
    NeuronMaxout, NeuronPAU, NeuronPELU, NeuronPReLU, NeuronReLU,
    NeuronSReLU, NeuronSoftExp, NeuronSwish,
)

ACTIVATIONS = (
    NeuronPReLU, NeuronSwish, NeuronPELU, NeuronSReLU, NeuronAPL,
    NeuronMaxout, NeuronPAU, NeuronKAF, NeuronACON, NeuronSoftExp,
)


class PrinceModel(Module):
    def __init__(self, input_size, hidden_units, activation):
        self.layers = (
            Layer(input_size, hidden_units, activation),
            Layer(hidden_units, 1, NeuronLinear),
        )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def hidden_units_below_target(input_size, target, activation):
    # calculate the cost contributed by each hidden unit for the given activation
    cost = len(PrinceModel(input_size, 1, activation).parameters()) - 1
    # P=1+H⋅cost
    # largest number of hidden units whose parameter count is strictly below target
    return max(1, (target - 2) // cost)


if __name__ == "__main__":
    with open("weird_functions.json") as file:
        input_sizes = sorted({
            len(ast.parse(item["numpy"], mode="eval").body.args.args)
            for item in json.load(file)
        })

    for input_size in input_sizes:
        target = len(PrinceModel(input_size, 16, NeuronReLU).parameters())
        print(f"\ninput={input_size}, ReLU target={target}")
        for activation in ACTIVATIONS:
            hidden = hidden_units_below_target(input_size, target, activation)
            actual = len(PrinceModel(input_size, hidden, activation).parameters())
            print(f"{activation.__name__:16} hidden={hidden:2} params={actual:2} delta={actual-target:+d}")
