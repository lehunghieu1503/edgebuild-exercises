"""
layers_04_non_pytorch.py
The layers that go with convolution, from scratch in pure Python (no NumPy, no PyTorch).
Same structure as layers_04.cpp.

  ReLU      : negatives become 0
  MaxPool   : shrink the image, keep the largest value of every 2x2 block
  BatchNorm : rescale every channel so values stay in a sane range
  Linear    : (fully-connected) turn the features into one score per class
  CNN       : blocks of "conv -> BatchNorm -> ReLU -> pool", then Linear

Run:  python layers_04_non_pytorch.py
"""

from math import sqrt


# ---------------------------------------------------------------------------
# 1. Tensor (CHW) and the weights of each layer type
# ---------------------------------------------------------------------------
class Tensor:
    def __init__(self, C, H, W, data=None):
        self.C, self.H, self.W = C, H, W
        self.data = list(data) if data is not None else [0.0] * (C * H * W)

    def at(self, c, h, w):
        return self.data[(c * self.H + h) * self.W + w]

    def set(self, c, h, w, value):
        self.data[(c * self.H + h) * self.W + w] = value


class Lcg:
    """Tiny deterministic random generator, so C++ and Python produce the same numbers."""

    def __init__(self, seed):
        self.state = seed

    def next(self):  # uniform in [-0.5, 0.5)
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / 16777216.0 - 0.5

    def fill(self, n, offset=0.0):
        return [offset + self.next() for _ in range(n)]


class Conv:
    def __init__(self, c_out, c_in, k):
        self.c_out, self.c_in, self.k = c_out, c_in, k
        self.weight = [0.0] * (c_out * c_in * k * k)  # (C_out, C_in, k, k)
        self.bias = [0.0] * c_out

    def num_params(self):
        return len(self.weight) + len(self.bias)


class BatchNorm:
    """
    Four numbers per channel. gamma/beta are learned like any weight;
    mean/var are statistics measured on the training data.
    """

    def __init__(self, channels):
        self.gamma, self.beta = [1.0] * channels, [0.0] * channels
        self.mean, self.var = [0.0] * channels, [1.0] * channels

    def num_params(self):
        return len(self.gamma) + len(self.beta)


class Linear:
    def __init__(self, n_in, n_out):
        self.n_in, self.n_out = n_in, n_out
        self.weight = [0.0] * (n_in * n_out)  # (out, in)
        self.bias = [0.0] * n_out

    def num_params(self):
        return len(self.weight) + len(self.bias)


# ---------------------------------------------------------------------------
# 2. The layers
# ---------------------------------------------------------------------------
def relu(x):
    """The only non-linear step. Without it, stacked layers collapse into one straight line."""
    return Tensor(x.C, x.H, x.W, [max(v, 0.0) for v in x.data])


def maxpool2d(x, k=2, stride=2):
    """One output per k x k block = the largest value in that block. No weights."""
    y = Tensor(x.C, (x.H - k) // stride + 1, (x.W - k) // stride + 1)
    for c in range(y.C):
        for oh in range(y.H):
            for ow in range(y.W):
                best = x.at(c, oh * stride, ow * stride)
                for kh in range(k):
                    for kw in range(k):
                        best = max(best, x.at(c, oh * stride + kh, ow * stride + kw))
                y.set(c, oh, ow, best)
    return y


def batchnorm(x, bn, eps=1e-5):
    """
    At inference: y = gamma * (x - mean) / sqrt(var + eps) + beta, per channel.
    All four numbers are constants here, so it is just "x * scale + shift".
    (That is why inference engines can fold it into the conv in front of it.)
    """
    y = Tensor(x.C, x.H, x.W, x.data)
    plane = x.H * x.W
    for c in range(x.C):
        scale = bn.gamma[c] / sqrt(bn.var[c] + eps)
        shift = bn.beta[c] - bn.mean[c] * scale
        for i in range(c * plane, (c + 1) * plane):
            y.data[i] = y.data[i] * scale + shift
    return y


def linear(x, fc):
    """y[o] = sum_i weight[o][i] * x[i] + bias[o]   (same as 02_layer_weight)"""
    y = []
    for o in range(fc.n_out):
        total = fc.bias[o]
        for i in range(fc.n_in):
            total += fc.weight[o * fc.n_in + i] * x[i]
        y.append(total)
    return y


def conv2d(x, conv, stride, pad):
    """Convolution with zero padding (explained step by step in 03_convolution)"""
    k = conv.k
    y = Tensor(conv.c_out, (x.H + 2 * pad - k) // stride + 1, (x.W + 2 * pad - k) // stride + 1)
    for co in range(conv.c_out):
        for oh in range(y.H):
            for ow in range(y.W):
                total = conv.bias[co]
                for ci in range(conv.c_in):
                    for kh in range(k):
                        for kw in range(k):
                            ih = oh * stride + kh - pad
                            iw = ow * stride + kw - pad
                            if ih < 0 or ih >= x.H or iw < 0 or iw >= x.W:
                                continue
                            total += (x.at(ci, ih, iw)
                                      * conv.weight[((co * conv.c_in + ci) * k + kh) * k + kw])
                y.set(co, oh, ow, total)
    return y


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def print_values(name, values, fmt="+.3f"):
    print(f"  {name} [ " + " ".join(f"{v:{fmt}}" for v in values) + " ]")


def print_channel(title, t):
    print(f"  {title}  ({t.H} x {t.W})")
    for h in range(t.H):
        print("   " + "".join(f"{t.at(0, h, w):3.0f}" for w in range(t.W)))


def print_row(name, t, params, note=""):
    print(f"  {name:<16} {f'({t.C}, {t.H}, {t.W})':<12} {params:6d}{note}")


def main():
    # (a) ReLU
    print("(a) ReLU: negatives become 0")
    r = Tensor(1, 1, 5, [-2.0, -0.5, 0.0, 1.5, 3.0])
    print_values("in ", r.data, "+.1f")
    print_values("out", relu(r).data, "+.1f")
    print()

    # (b) MaxPool
    print("(b) MaxPool 2x2: keep the largest value of every 2x2 block")
    p = Tensor(1, 4, 4, [1, 3, 2, 0,
                         4, 2, 1, 1,
                         0, 1, 5, 6,
                         2, 0, 7, 8])
    print_channel("in ", p)
    print_channel("out", maxpool2d(p))
    print("  16 numbers -> 4 numbers: the next layer has 4x less work\n")

    # (c) BatchNorm: two channels on wildly different scales
    print("(c) BatchNorm: put every channel on the same scale")
    b = Tensor(2, 2, 2, [1000, 2000, 3000, 4000,   # channel 0
                         1, 2, 3, 4])              # channel 1
    bn = BatchNorm(2)
    bn.mean = [2500.0, 2.5]         # mean of each channel
    bn.var = [1250000.0, 1.25]      # variance of each channel
    normed = batchnorm(b, bn)
    for c in range(2):
        before = " ".join(f"{v:.0f}" for v in b.data[c * 4:c * 4 + 4])
        after = " ".join(f"{v:+.3f}" for v in normed.data[c * 4:c * 4 + 4])
        print(f"  channel {c}  before [ {before} ]  after [ {after} ]")
    print("  per channel: 2 learned weights (gamma, beta) + 2 statistics (mean, var)\n")

    # (d) A CNN: two "conv -> BatchNorm -> ReLU -> pool" blocks, then Linear
    print("(d) A CNN: 1x8x8 image -> 3 class scores")
    rng = Lcg(7)
    image = Tensor(1, 8, 8, rng.fill(64))

    conv1, bn1 = Conv(4, 1, 3), BatchNorm(4)
    conv2, bn2 = Conv(8, 4, 3), BatchNorm(8)
    fc = Linear(8 * 2 * 2, 3)
    for conv, bn in ((conv1, bn1), (conv2, bn2)):
        conv.weight = rng.fill(len(conv.weight))
        conv.bias = rng.fill(conv.c_out)
        bn.gamma = rng.fill(conv.c_out, 1.0)
        bn.beta = rng.fill(conv.c_out)
        bn.mean = rng.fill(conv.c_out)
        bn.var = rng.fill(conv.c_out, 1.0)
    fc.weight = rng.fill(len(fc.weight))
    fc.bias = rng.fill(fc.n_out)

    print("  layer            output shape params")
    x = image
    print_row("input", x, 0)
    x = conv2d(x, conv1, 1, 1)
    print_row("conv1 1->4 3x3", x, conv1.num_params())
    x = batchnorm(x, bn1)
    print_row("bn1", x, bn1.num_params(), "  (+8 statistics)")
    x = relu(x)
    print_row("relu1", x, 0)
    x = maxpool2d(x)
    print_row("pool1 2x2", x, 0, "  (image halves)")
    x = conv2d(x, conv2, 1, 1)
    print_row("conv2 4->8 3x3", x, conv2.num_params())
    x = batchnorm(x, bn2)
    print_row("bn2", x, bn2.num_params(), "  (+16 statistics)")
    x = relu(x)
    print_row("relu2", x, 0)
    x = maxpool2d(x)
    print_row("pool2 2x2", x, 0, "  (image halves)")

    # Flatten: (8, 2, 2) -> 32 numbers. Nothing is computed, the buffer is just read as 1D.
    scores = linear(x.data, fc)
    print(f"  {'flatten':<16} {'(32)':<12} {0:6d}")
    print(f"  {'fc 32->3':<16} {'(3)':<12} {fc.num_params():6d}")
    total = sum(l.num_params() for l in (conv1, bn1, conv2, bn2, fc))
    print(f"  total params = {total}")
    print_values("scores", scores, "+.4f")


if __name__ == "__main__":
    main()
