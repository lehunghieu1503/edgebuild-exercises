"""
layer_02_non_pytorch.py
Layers and weights from scratch in pure Python (no NumPy, no PyTorch).
Same structure as layer_02.cpp: it loads the model file that layer_02_pytorch.py
trained and wrote, and runs it with plain loops.

  layer      = a function "tensor in -> tensor out" + the numbers it computes with
  weights    = those numbers (learned from data, nobody types them)
  network    = a chain of layers
  model file = the chain's configuration + the value of every weight

Run:  python layer_02_non_pytorch.py [path/to/line_model.txt]
"""

import random
import sys
from pathlib import Path

MODEL_FILE = Path(__file__).resolve().parent.parent / "line_model.txt"
CLASSES = ["vertical", "horizontal"]


# ---------------------------------------------------------------------------
# 1. Layers: take a tensor, compute, return another tensor.
#    A tensor here is just a flat list of floats (see 01_tensor/Python/non_pytorch.py for shape/strides).
# ---------------------------------------------------------------------------
class Linear:
    """Fully-connected layer: y[o] = sum_i weight[o][i] * x[i] + bias[o]"""

    type = "linear"

    def __init__(self, name, n_in, n_out):
        self.name = name
        self.n_in, self.n_out = n_in, n_out   # configuration
        self.weight = [0.0] * (n_in * n_out)  # row o = weights of output o
        self.bias = [0.0] * n_out

    def forward(self, x):
        if len(x) != self.n_in:
            raise ValueError(f"{self.name}: expected {self.n_in} inputs, got {len(x)}")
        y = []
        for o in range(self.n_out):
            total = self.bias[o]
            for i in range(self.n_in):
                total += self.weight[o * self.n_in + i] * x[i]
            y.append(total)
        return y

    def num_params(self):
        return len(self.weight) + len(self.bias)


class ReLU:
    """Negatives become 0. A layer with NO weights: it only computes."""

    type = "relu"

    def __init__(self, name):
        self.name = name

    def forward(self, x):
        return [max(v, 0.0) for v in x]

    def num_params(self):
        return 0


# ---------------------------------------------------------------------------
# 2. Network: a chain of layers. The output of one is the input of the next.
# ---------------------------------------------------------------------------
def print_tensor(name, t):
    print(f"  {name:<6} shape=({len(t)})  [ " + " ".join(f"{v:+.3f}" for v in t) + " ]")


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, trace=False):
        if trace:
            print_tensor("input", x)
        for layer in self.layers:
            x = layer.forward(x)
            if trace:
                print_tensor(layer.name, x)
        return x

    def print_summary(self):
        print("  #  name   type    in -> out  params")
        total = 0
        for i, l in enumerate(self.layers, 1):
            if isinstance(l, Linear):
                io = f"{l.n_in} -> {l.n_out}"
                note = f"  ({l.n_out}*{l.n_in} weights + {l.n_out} biases)"
            else:
                io, note = "same", "  (no weights: it only computes)"
            print(f"  {i}  {l.name:<6} {l.type:<7} {io:<10} {l.num_params()}{note}")
            total += l.num_params()
        print(f"  total params = {total}  (= {total * 4} bytes as float32)")


# ---------------------------------------------------------------------------
# 3. Two ways to get a Model: random weights, or weights from a model file
# ---------------------------------------------------------------------------
def make_untrained_model(seed):
    """Same configuration as the file, but the weights are random noise."""
    rng = random.Random(seed)
    fc1, fc2 = Linear("fc1", 9, 6), Linear("fc2", 6, 2)
    for fc in (fc1, fc2):
        fc.weight = [rng.uniform(-0.5, 0.5) for _ in fc.weight]
        fc.bias = [rng.uniform(-0.5, 0.5) for _ in fc.bias]
    return Model([fc1, ReLU("relu1"), fc2])


def load_model(path):
    """
    The file says which layers exist, in what order, and the value of every weight.
    Nothing about the network is hard-coded here: change the file, get another network.
    """
    with open(path) as f:
        # Drop '#' comments; what is left is one whitespace-separated token stream
        tokens = iter(" ".join(line.split("#")[0] for line in f).split())

    def expect(word):
        tok = next(tokens, None)
        if tok != word:
            raise ValueError(f"model file: expected '{word}', got '{tok}'")

    def read_floats(n):
        try:
            return [float(next(tokens)) for _ in range(n)]
        except (StopIteration, RuntimeError):
            raise ValueError("model file: not enough numbers") from None

    expect("layers")
    layers = []
    for _ in range(int(next(tokens))):
        kind, name = next(tokens), next(tokens)
        if kind == "linear":
            fc = Linear(name, int(next(tokens)), int(next(tokens)))
            expect("weight")
            fc.weight = read_floats(fc.n_in * fc.n_out)
            expect("bias")
            fc.bias = read_floats(fc.n_out)
            layers.append(fc)
        elif kind == "relu":
            layers.append(ReLU(name))
        else:
            raise ValueError(f"model file: unknown layer type '{kind}'")
    return Model(layers)


# ---------------------------------------------------------------------------
# 4. Test data: six 3x3 images, flattened to 9 numbers
# ---------------------------------------------------------------------------
def line_images():
    samples = []
    for k in range(3):
        v, h = [0.0] * 9, [0.0] * 9
        for j in range(3):
            v[j * 3 + k] = 1.0  # column k is lit
            h[k * 3 + j] = 1.0  # row k is lit
        samples += [(v, 0), (h, 1)]
    return samples


def evaluate(model, samples):
    correct = 0
    for image, label in samples:
        scores = model.forward(image)
        pred = int(scores[1] > scores[0])  # the bigger score wins
        correct += pred == label
        rows = ["".join("#" if p > 0.5 else "." for p in image[r * 3:r * 3 + 3]) for r in range(3)]
        verdict = "OK" if pred == label else f"WRONG (it is {CLASSES[label]})"
        print(f"  {rows[0]}   scores: vertical={scores[0]:+.3f}  horizontal={scores[1]:+.3f}")
        print(f"  {rows[1]}   -> {CLASSES[pred]}  {verdict}")
        print(f"  {rows[2]}")
    print(f"  correct: {correct}/{len(samples)}\n")


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------
def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_FILE

    # (a) ONE layer, weights typed by hand so the arithmetic is visible
    print("(a) One layer: Linear(3 -> 2), hand-typed weights")
    layer = Linear("demo", 3, 2)
    layer.weight = [1.0, 0.0, -1.0,   # output 0
                    0.5, 0.5, 0.5]    # output 1
    layer.bias = [0.0, 1.0]
    x = [2.0, 3.0, 4.0]
    y = layer.forward(x)
    print_tensor("x", x)
    print(f"  y[0] = 1.0*2 + 0.0*3 + -1.0*4 + 0.0 = {y[0]:+.1f}")
    print(f"  y[1] = 0.5*2 + 0.5*3 +  0.5*4 + 1.0 = {y[1]:+.1f}")
    print_tensor("y", y)
    print_tensor("relu", ReLU("relu").forward(y))
    print()

    samples = line_images()

    # (b) The right layers but random weights: the network knows nothing
    print("(b) Untrained network (random weights)")
    untrained = make_untrained_model(1)
    untrained.print_summary()
    evaluate(untrained, samples)

    # (c) Load the model file: configuration + weights learned by PyTorch
    print(f"(c) Network loaded from {model_path}")
    model = load_model(model_path)
    model.print_summary()
    evaluate(model, samples)

    # (d) Follow one image through the chain: every layer is tensor in, tensor out
    print("(d) One image through the chain, layer by layer")
    out = model.forward(samples[0][0], trace=True)
    print(f"  -> {CLASSES[int(out[1] > out[0])]}\n")

    # (b) and (c) ran the exact same code. Only the 74 numbers differ.
    # ResNet-18 is the same idea with 11,689,512 numbers (~46.8 MB as float32).


if __name__ == "__main__":
    main()
