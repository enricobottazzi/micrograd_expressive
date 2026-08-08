"""Bench: ReLU (16 hidden units) vs parameter-matched adaptive activations
on the weird functions. Heatmap pixel = per-function rank of the activation's
mean test loss over seeds (green = best, red = worst)."""
import ast
import json
import random
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from micrograd.nn import NeuronReLU
from micrograd.utils import loss
from parameter_matching import ACTIVATIONS, PrinceModel, hidden_units_below_target

ALL_ACTIVATIONS = (NeuronReLU,) + ACTIVATIONS

# all-NaN cells (every seed diverged) are expected; keep the log readable
warnings.filterwarnings("ignore", message="Mean of empty slice")

def run(task):
    """Train one (dataset, activation, seed) combo; return final test loss."""
    X_train, y_train, X_test, y_test, activation, hidden, seed, steps = task
    random.seed(seed)     # model init
    np.random.seed(seed)  # batch sampling
    model = PrinceModel(X_train.shape[1], hidden, activation)
    try:
        for _ in range(steps):
            total = loss(model, X_train, y_train, batch_size=32)
            model.zero_grad()
            total.backward()
            for p in model.parameters():  # sgd
                p.data -= 0.01 * p.grad
        return loss(model, X_test, y_test).data
    except (ValueError, OverflowError, ZeroDivisionError):
        return float("nan")  # diverged run (e.g. SoftExp log-branch domain error)


def bench(steps=1000, seeds=range(10), out="bench_heatmap"):
    N = 10000
    np.random.seed(1337)
    ri = np.random.permutation(N)  # train/test split, shared by every run
    n_train = int(0.8 * N)

    with open("weird_functions.json") as f:
        fns = [(i, eval(item["numpy"]),
                len(ast.parse(item["numpy"], mode="eval").body.args.args))
               for i, item in enumerate(json.load(f))]

    # one X per input size, sampled once and shared by every run:
    # 1-D keeps the linspace grid, higher dims sample [-10, 10]^D uniformly
    dims = sorted({d for _, _, d in fns})
    Xs = {d: np.linspace(-10, 10, N).reshape(-1, 1) if d == 1
          else np.random.uniform(-10, 10, (N, d)) for d in dims}

    # per input size, ReLU (16 hidden) fixes the parameter budget; other
    # activations get the widest hidden layer that stays below it
    targets = {d: len(PrinceModel(d, 16, NeuronReLU).parameters()) for d in dims}
    width = lambda d, a: (16 if a is NeuronReLU
                          else hidden_units_below_target(d, targets[d], a))

    tasks = []
    for _, fn, d in fns:
        X, y = Xs[d], fn(*Xs[d].T)  # one column per argument
        data = (X[ri[:n_train]], y[ri[:n_train]], X[ri[n_train:]], y[ri[n_train:]])
        tasks += [(*data, a, width(d, a), s, steps)
                  for a in ALL_ACTIVATIONS for s in seeds]

    print(f"{len(fns)} functions x {len(ALL_ACTIVATIONS)} activations x "
          f"{len(seeds)} seeds ({steps} steps) -> {len(tasks)} runs")

    with ProcessPoolExecutor() as ex:
        losses = []
        for l in ex.map(run, tasks, chunksize=1):  # results arrive in task order
            losses.append(l)
            if len(losses) % len(seeds):  # print once per completed (fn, act) cell
                continue
            cell = len(losses) // len(seeds) - 1
            i, _, d = fns[cell // len(ALL_ACTIVATIONS)]
            act = ALL_ACTIVATIONS[cell % len(ALL_ACTIVATIONS)]
            print(f"fn {i:3} (d={d}) {act.__name__.replace('Neuron', ''):8} "
                  f"mean loss {np.nanmean(losses[-len(seeds):]):9.4f} "
                  f"[{len(losses)}/{len(tasks)}]", flush=True)

    # rows = function, cols = activation, value = mean test loss over seeds
    results = np.nanmean(
        np.array(losses).reshape(len(fns), len(ALL_ACTIVATIONS), len(seeds)), axis=2)
    np.save(f"{out}.npy", results)  # raw losses, in case a different plot is needed

    # rank activations within each function: 1 = lowest loss (NaNs rank last)
    ranks = results.argsort(axis=1).argsort(axis=1) + 1

    plt.figure(figsize=(7, 0.15 * len(fns) + 2))
    plt.imshow(ranks, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=ranks.shape[1])
    plt.colorbar(label=f"rank (1 = best of {ranks.shape[1]})")
    plt.xticks(range(len(ALL_ACTIVATIONS)),
               [a.__name__.replace("Neuron", "") for a in ALL_ACTIVATIONS],
               rotation=45, ha="right")
    plt.yticks(range(len(fns)), [i for i, _, _ in fns], fontsize=5)
    plt.xlabel("activation")
    plt.ylabel("function index")
    plt.tight_layout()
    plt.savefig(f"{out}.png", dpi=200)
    plt.close()
    return results


if __name__ == "__main__":
    bench()
